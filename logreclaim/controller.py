"""Orchestration: scan, plan, snapshot, revalidate and execute.

The web layer is intentionally thin; all decisions live here so they can be
tested without starting a server.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import agents, procfs, scan as scan_mod
from .config import Config
from .planner import (
    Candidate,
    LogRecord,
    Plan,
    compute_plan,
    released_logs,
)

logger = logging.getLogger(__name__)

# How long to let writers finish a write and close old descriptors.
SETTLE_SECONDS = 0.5

# Background host-scan lifecycle states.
HOST_SCAN_DISABLED = "disabled"  # SCAN_SCOPE != host
HOST_SCAN_IDLE = "idle"          # host scope, no scan for the current epoch yet
HOST_SCAN_RUNNING = "running"    # background full-host scan in progress
HOST_SCAN_READY = "ready"        # full-host table available for the current epoch
HOST_SCAN_FAILED = "failed"      # last background scan raised; first screen retained
HOST_SCAN_LIVE = "live"          # view came from a synchronous on-demand full scan


@dataclass(frozen=True)
class HostScanStatus:
    """Snapshot of the background full-host scan state for the UI."""

    state: str
    table_epoch: int
    attempt: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    host_scope: bool = False

    @property
    def is_running(self) -> bool:
        return self.state == HOST_SCAN_RUNNING

    @property
    def is_failed(self) -> bool:
        return self.state == HOST_SCAN_FAILED

    @property
    def table_is_host_wide(self) -> bool:
        """True only when the rendered table already covers every visible process."""
        return self.state in (HOST_SCAN_READY, HOST_SCAN_LIVE)


@dataclass
class LogSnapshot:
    key: procfs.FileKey
    path: str
    blocks: int
    holders: Dict[procfs.ProcId, List[int]]
    inode_links: int

    @property
    def allocated(self) -> int:
        return self.blocks * 512

    @classmethod
    def of(cls, log: LogRecord) -> "LogSnapshot":
        return cls(
            key=log.key,
            path=log.path,
            blocks=log.blocks,
            holders={p: list(fds) for p, fds in log.holders.items()},
            inode_links=log.inode_links,
        )


@dataclass
class PlanSnapshot:
    token: str
    created: float
    target_bytes: int
    target_mib: float
    selected: List[procfs.ProcId]
    covered: List[LogSnapshot]
    plan: Plan


@dataclass
class ProcResult:
    process: procfs.ProcId
    name: str
    ok: bool
    detail: str
    reopened: List[dict] = field(default_factory=list)


@dataclass
class ExecutionReport:
    executed: bool
    reasons: List[str] = field(default_factory=list)
    message: str = ""
    results: List[ProcResult] = field(default_factory=list)
    freed_bytes: int = 0
    target_bytes: int = 0
    released: List[LogSnapshot] = field(default_factory=list)
    remaining: List[LogSnapshot] = field(default_factory=list)
    remaining_holders: Dict[procfs.FileKey, List[str]] = field(default_factory=dict)
    checked_at: float = 0.0


@dataclass
class ScanView:
    scan: scan_mod.Scan
    agents_live: List[agents.AgentView]
    agents_all: List[agents.AgentView]
    controllable: List[procfs.ProcId]
    plan_candidates: List[Candidate]
    host_status: Optional[HostScanStatus] = None

    @property
    def logs(self) -> List[LogRecord]:
        return self.scan.logs

    def candidates(self) -> List[Candidate]:
        return self.plan_candidates

    def display(self, proc_id: procfs.ProcId) -> str:
        name = self.scan.display.get(proc_id)
        return f"{proc_id.pid} · {name}" if name else f"{proc_id.pid} · 未知进程"

    def agent_for(self, proc_id: procfs.ProcId) -> Optional[agents.AgentView]:
        for view in self.agents_live:
            if view.ident == proc_id:
                return view
        return None

    def holder_label(self, proc_id: procfs.ProcId, with_state: bool = True) -> str:
        agent = self.agent_for(proc_id)
        base = self.display(proc_id)
        if agent is None and with_state:
            return f"{base}（非受控）"
        return base

    def is_controllable(self, proc_id: procfs.ProcId) -> bool:
        return any(v.ident == proc_id for v in self.agents_live)


class Controller:
    def __init__(self, config: Config):
        self.config = config
        self._plans: Dict[str, PlanSnapshot] = {}
        self._lock = threading.Lock()

        # -- background full-host scan machinery ----------------------------
        # The table epoch is a monotonic barrier: every mutating action
        # (rotate / delete / confirmed reclamation) bumps it, and a background
        # scan may only publish results for the epoch it started in.  Late
        # results of an older, possibly stuck, scan can never overwrite the
        # table a newer scan produced.
        self._host_lock = threading.Lock()
        self._table_epoch = 0
        self._host_attempt = 0
        self._host_state = HOST_SCAN_IDLE
        self._host_view: Optional[ScanView] = None
        self._host_view_epoch: int = -1
        self._host_error = ""
        self._host_started_at = 0.0
        self._host_finished_at = 0.0
        self._host_thread: Optional[threading.Thread] = None
        # Test / simulation hook: when set, a background worker blocks here
        # before scanning.  It is deliberately not consulted by synchronous
        # scans, so planning and confirmation stay real-time.
        self.host_scan_gate: Optional[threading.Event] = None
        # Artificial slowdown for background scans only, in seconds.
        self.host_scan_delay: float = 0.0

    # -- scanning ----------------------------------------------------------
    def _build_view(self, only_pids: Optional[List[int]] = None) -> ScanView:
        """Build a ScanView from a fresh /proc pass.

        ``only_pids`` restricts the /proc pass to registered writer PIDs (the
        first screen).  When omitted the scope follows ``SCAN_SCOPE``: the
        registered writers only, or every visible process on the host.
        """
        views_all = agents.verify_registry(self.config.state_dir)
        live = [v for v in views_all if v.alive]
        agent_names = {v.ident: v.info.name for v in live}
        agent_map = {v.ident: v.info for v in live}

        if only_pids is None and self.config.scan_scope == "agents":
            only_pids = [v.info.pid for v in live] or [-1]
        base = scan_mod.collect(
            scope_uid=None,
            log_roots=self.config.log_roots,
            only_pids=only_pids,
        )
        # Prefer the demo writer's registered name over its argv.
        for proc_id, name in agent_names.items():
            base.display[proc_id] = name

        controllable = list(agent_map)
        allowed = set(controllable)
        plan_candidates = [c for c in base.candidates if c.process in allowed]
        for cand in plan_candidates:
            if cand.process in agent_names:
                cand.display = agent_names[cand.process]

        # Candidates that are live but currently hold no deleted log are still
        # reported in the UI but cannot contribute to a cover.
        return ScanView(
            scan=base,
            agents_live=live,
            agents_all=views_all,
            controllable=controllable,
            plan_candidates=plan_candidates,
        )

    def scan(self) -> ScanView:
        """Synchronous, real-time scan used by planning and confirmation."""
        view = self._build_view()
        view.host_status = self.live_status()
        return view

    def quick_view(self) -> ScanView:
        """First screen in host scope: only the registered writer processes."""
        view = self._build_view(only_pids=self._registered_pids())
        view.host_status = self.host_scan_status()
        return view

    def _registered_pids(self) -> List[int]:
        live = [v for v in agents.verify_registry(self.config.state_dir) if v.alive]
        return [v.info.pid for v in live] or [-1]

    # -- background full-host scan ----------------------------------------
    def index_view(self) -> ScanView:
        """View for the home page.

        In host scope the page returns immediately with the controlled-writer
        first screen while a full-host scan runs in the background; once a
        scan of the current table epoch finishes, the complete table is shown.
        """
        if self.config.scan_scope != "host":
            return self.scan()
        with self._host_lock:
            status = self._status_locked()
            if (
                status.state == HOST_SCAN_READY
                and self._host_view is not None
                and self._host_view_epoch == self._table_epoch
            ):
                view = self._host_view
                view.host_status = status
                return view
            if status.state == HOST_SCAN_IDLE:
                # A failure is only recovered by an explicit retry, so the
                # first screen stays stable instead of flapping on refresh.
                self._ensure_background_scan_locked()
        return self.quick_view()

    def _ensure_background_scan_locked(self) -> None:
        """Start a background scan for the current epoch unless one is running."""
        if self._host_state == HOST_SCAN_RUNNING:
            return
        self._start_background_scan_locked()

    def _start_background_scan_locked(self) -> None:
        self._table_epoch  # read current epoch under the lock
        self._host_attempt += 1
        epoch = self._table_epoch
        attempt = self._host_attempt
        self._host_state = HOST_SCAN_RUNNING
        self._host_error = ""
        self._host_started_at = time.time()
        self._host_finished_at = 0.0
        thread = threading.Thread(
            target=self._run_host_scan,
            name=f"host-scan-e{epoch}-a{attempt}",
            args=(epoch, attempt),
            daemon=True,
        )
        self._host_thread = thread
        thread.start()

    def _run_host_scan(self, epoch: int, attempt: int) -> None:
        """Worker: one full-host scan. Results publish only if still current."""
        gate = self.host_scan_gate
        if gate is not None:
            gate.wait()
        if self.host_scan_delay:
            time.sleep(self.host_scan_delay)
        with self._host_lock:
            # A newer epoch/attempt may already have superseded this worker
            # while it was blocked on the gate; do not even start the scan.
            if not self._still_current_locked(epoch, attempt):
                logger.info(
                    "skip superseded host scan: epoch %s attempt %s, "
                    "current epoch %s attempt %s",
                    epoch, attempt, self._table_epoch, self._host_attempt,
                )
                return
        try:
            view = self._build_view()
        except Exception as exc:  # the home page must survive a failing scan
            logger.exception("background host scan failed")
            with self._host_lock:
                if self._still_current_locked(epoch, attempt):
                    self._host_state = HOST_SCAN_FAILED
                    self._host_error = str(exc) or exc.__class__.__name__
                    self._host_finished_at = time.time()
            return
        with self._host_lock:
            # Mutation barrier: a newer table epoch (or a retry) supersedes
            # this worker, so its late result is discarded.
            if not self._still_current_locked(epoch, attempt):
                logger.info(
                    "discard stale host scan: result epoch %s attempt %s, "
                    "current epoch %s attempt %s",
                    epoch, attempt, self._table_epoch, self._host_attempt,
                )
                return
            view.host_status = self._status_locked()
            self._host_view = view
            self._host_view_epoch = epoch
            self._host_state = HOST_SCAN_READY
            self._host_error = ""
            self._host_finished_at = time.time()

    def _still_current_locked(self, epoch: int, attempt: int) -> bool:
        return (
            self._table_epoch == epoch
            and self._host_attempt == attempt
            and self._host_state == HOST_SCAN_RUNNING
        )

    def retry_host_scan(self) -> HostScanStatus:
        """User-requested retry: always start a fresh background full scan."""
        with self._host_lock:
            if self.config.scan_scope != "host":
                return self._status_locked()
            self._start_background_scan_locked()
            return self._status_locked()

    def bump_table_epoch(self) -> int:
        """Invalidate the published table and any in-flight background scan.

        Called after every mutating action so the next full-host result the
        page accepts can only come from a scan that began afterwards.
        """
        with self._host_lock:
            self._table_epoch += 1
            self._host_attempt += 1
            self._host_view = None
            self._host_view_epoch = -1
            self._host_state = HOST_SCAN_IDLE
            self._host_error = ""
            self._host_finished_at = 0.0
            return self._table_epoch

    def host_scan_status(self) -> HostScanStatus:
        with self._host_lock:
            return self._status_locked()

    def _status_locked(self) -> HostScanStatus:
        if self.config.scan_scope != "host":
            return HostScanStatus(
                state=HOST_SCAN_DISABLED,
                table_epoch=self._table_epoch,
                host_scope=False,
            )
        return HostScanStatus(
            state=self._host_state,
            table_epoch=self._table_epoch,
            attempt=self._host_attempt,
            started_at=self._host_started_at,
            finished_at=self._host_finished_at,
            error=self._host_error,
            host_scope=True,
        )

    def live_status(self) -> HostScanStatus:
        """Status stamp for a view produced by a synchronous on-demand scan."""
        status = self.host_scan_status()
        if not status.host_scope:
            return status
        return HostScanStatus(
            state=HOST_SCAN_LIVE,
            table_epoch=status.table_epoch,
            attempt=status.attempt,
            started_at=status.started_at,
            finished_at=time.time(),
            error="",
            host_scope=True,
        )

    # -- planning ----------------------------------------------------------
    def make_plan(self, view: ScanView, target_bytes: int) -> Plan:
        return compute_plan(view.logs, view.candidates(), target_bytes)

    def snapshot(self, view: ScanView, plan: Plan, target_mib: float) -> PlanSnapshot:
        token = secrets.token_urlsafe(18)
        snap = PlanSnapshot(
            token=token,
            created=time.time(),
            target_bytes=plan.target_bytes,
            target_mib=target_mib,
            selected=[c.process for c in plan.selected],
            covered=[LogSnapshot.of(log) for log in plan.covered],
            plan=plan,
        )
        with self._lock:
            self._prune_locked()
            self._plans[token] = snap
        return snap

    def get_snapshot(self, token: str) -> Optional[PlanSnapshot]:
        with self._lock:
            snap = self._plans.get(token)
        if snap is None:
            return None
        if time.time() - snap.created > self.config.plan_ttl_seconds:
            with self._lock:
                self._plans.pop(token, None)
            return None
        return snap

    def take_snapshot(self, token: str) -> Optional[PlanSnapshot]:
        """One-shot claim: a confirmation token can be executed at most once."""
        with self._lock:
            snap = self._plans.pop(token, None)
        if snap is None:
            return None
        if time.time() - snap.created > self.config.plan_ttl_seconds:
            return None
        return snap

    def drop_snapshot(self, token: str) -> None:
        with self._lock:
            self._plans.pop(token, None)

    def _prune_locked(self) -> None:
        cutoff = time.time() - self.config.plan_ttl_seconds
        for token in [t for t, s in self._plans.items() if s.created < cutoff]:
            self._plans.pop(token, None)

    # -- validation --------------------------------------------------------
    def validate(
        self, snap: PlanSnapshot, view: ScanView
    ) -> Tuple[List[str], Optional[Plan]]:
        """Check a saved plan against a fresh scan.

        Returns (reasons, fresh_plan).  Non-empty reasons mean: recompute,
        nothing was executed.
        """
        reasons: List[str] = []
        current = {log.key: log for log in view.logs}

        for previous in snap.covered:
            now = current.get(previous.key)
            if now is None:
                reasons.append(
                    f"旧日志 {previous.path or previous.key.label} "
                    f"（inode {previous.key.label}）已不再被打开。"
                )
                continue
            if now.path != previous.path:
                reasons.append(
                    f"旧日志 inode {previous.key.label} 的路径已从 "
                    f"{previous.path} 变为 {now.path}。"
                )
            if set(now.holders) != set(previous.holders):
                reasons.append(
                    f"旧日志 {previous.path or previous.key.label} 的持有者已变化。"
                )
            if now.inode_links != previous.inode_links:
                reasons.append(
                    f"旧日志 {previous.path or previous.key.label} 的链接数已变化，"
                    "可能被重新链接。"
                )

        fresh_ids = {c.process for c in view.candidates()}
        for proc_id in snap.selected:
            if proc_id not in fresh_ids:
                reasons.append(
                    f"计划中的进程 {proc_id.pid}（启动时间 {proc_id.starttime}）"
                    "已退出或不再持有已删除日志。"
                )

        # Does the original list still free enough, measured on fresh occupancy?
        selected = set(snap.selected)
        freed_now = sum(
            log.allocated for log in released_logs(view.logs, selected)
        )
        if freed_now < snap.target_bytes:
            reasons.append(
                f"原名单按当前占用只能释放 {freed_now} 字节，"
                f"低于目标 {snap.target_bytes} 字节。"
            )

        fresh_plan = compute_plan(
            view.logs, view.candidates(), snap.target_bytes
        )
        if not fresh_plan.feasible:
            reasons.append(
                "按当前占用无法达标：" + (fresh_plan.note or "无可用组合")
            )
        elif len(fresh_plan.selected) < len(snap.selected):
            reasons.append(
                f"按当前占用存在更少进程的名单（{len(fresh_plan.selected)} 个），"
                "原名单不再最少。"
            )
        return reasons, fresh_plan

    # -- execution ---------------------------------------------------------
    def execute(self, snap: PlanSnapshot) -> ExecutionReport:
        view = self.scan()
        reasons, _fresh = self.validate(snap, view)
        report = ExecutionReport(
            executed=False,
            target_bytes=snap.target_bytes,
            checked_at=time.time(),
        )
        if reasons:
            report.reasons = reasons
            report.message = "状态已变化，本次未执行，请重新计算。"
            self.drop_snapshot(snap.token)
            # The state changed before our eyes; any background table is now
            # stale, so force the next home-page result to be a fresh scan.
            self.bump_table_epoch()
            return report

        # Confirm-time occupancy for the report: every log the selected set
        # fully holds right now, not just the ones the plan happened to cover.
        before: Dict[procfs.FileKey, LogSnapshot] = {
            log.key: LogSnapshot.of(log)
            for log in released_logs(view.logs, set(snap.selected))
        }

        # Reopen the current logs of each selected process.
        for proc_id in snap.selected:
            agent = view.agent_for(proc_id)
            if agent is None:
                report.results.append(
                    ProcResult(
                        process=proc_id,
                        name=view.display(proc_id),
                        ok=False,
                        detail="进程已退出或不可控，仍沿用原句柄",
                    )
                )
                continue
            paths = list(agent.info.paths)
            try:
                response = agents.send_command(
                    self.config.state_dir,
                    agent.info.socket,
                    {"op": "reopen", "paths": paths},
                )
            except agents.AgentError as exc:
                report.results.append(
                    ProcResult(
                        process=proc_id,
                        name=agent.info.name,
                        ok=False,
                        detail=f"{exc}；仍沿用原句柄",
                    )
                )
                continue
            items = response.get("results") or []
            opened = [i for i in items if i.get("reopened")]
            failed = [i for i in items if not i.get("reopened")]
            detail = f"已重开 {len(opened)} 个日志文件"
            if failed:
                failures = "；".join(
                    f"{i.get('path')}（{i.get('reason', '失败')}）" for i in failed
                )
                detail += f"；失败 {len(failed)} 个，仍沿用原句柄：{failures}"
            report.results.append(
                ProcResult(
                    process=proc_id,
                    name=agent.info.name,
                    ok=bool(opened) and not failed,
                    detail=detail,
                    reopened=items,
                )
            )

        time.sleep(SETTLE_SECONDS)
        after = self.scan()
        after_map = {log.key: log for log in after.logs}

        # Recheck the original inodes, reporting with confirm-time occupancy.
        report.released = []
        report.remaining = []
        report.remaining_holders = {}
        for key, snapshot in before.items():
            still = after_map.get(key)
            if still is None:
                report.released.append(snapshot)
            else:
                report.remaining.append(snapshot)
                report.remaining_holders[key] = [
                    after.holder_label(proc_id)
                    for proc_id in sorted(still.holders)
                ]
        report.freed_bytes = sum(log.allocated for log in report.released)
        report.executed = True
        report.message = (
            f"已按名单重开日志。按确认时占用，确认释放 {report.freed_bytes} 字节。"
        )
        self.drop_snapshot(snap.token)
        # Reclamation changed holder relationships: the next full-host table
        # must come from a scan started strictly after this point.
        self.bump_table_epoch()
        return report

    # -- demo helpers ------------------------------------------------------
    def demo_rotate(self) -> List[Tuple[str, str]]:
        """Rename each live writer log to a .rot file; writers keep writing."""
        moved: List[Tuple[str, str]] = []
        stamp = int(time.time())
        paths = sorted(self._registered_paths())
        for path in paths:
            if not self._within_roots(path):
                continue
            if not os.path.isfile(path) or os.path.islink(path):
                continue
            target = f"{path}.{stamp}.rot"
            counter = 1
            while os.path.exists(target):
                target = f"{path}.{stamp}.{counter}.rot"
                counter += 1
            try:
                os.rename(path, target)
            except OSError:
                continue
            moved.append((path, target))
        if moved:
            # Rotation changes link state: invalidate any in-flight background
            # scan so its late result can never overwrite the post-action view.
            self.bump_table_epoch()
        return moved

    def demo_delete_old(self) -> List[Tuple[str, int]]:
        """Unlink rotated logs; writers holding them keep the space."""
        deleted: List[Tuple[str, int]] = []
        for root in self.config.log_roots:
            if not os.path.isdir(root):
                continue
            for entry in sorted(os.listdir(root)):
                if not entry.endswith(".rot"):
                    continue
                path = os.path.join(root, entry)
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                try:
                    size = os.stat(path).st_blocks * 512
                except OSError:
                    continue
                try:
                    os.unlink(path)
                except OSError:
                    continue
                deleted.append((path, size))
        if deleted:
            # Unlinking changes nlink / holder visibility of these inodes.
            self.bump_table_epoch()
        return deleted

    def _registered_paths(self):
        seen = set()
        for view in agents.verify_registry(self.config.state_dir):
            for path in view.info.paths:
                if path not in seen:
                    seen.add(path)
                    yield path

    def _within_roots(self, path: str) -> bool:
        real = os.path.realpath(path)
        for root in self.config.log_roots:
            real_root = os.path.realpath(root)
            if real == real_root or real.startswith(real_root + os.sep):
                return True
        return False

