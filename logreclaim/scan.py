"""Turn a live /proc snapshot into planner records."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import procfs
from .planner import Candidate, LogRecord, dedupe_logs


@dataclass
class Scan:
    processes: List[procfs.ProcessInfo]
    logs: List[LogRecord]
    candidates: List[Candidate]
    display: Dict[procfs.ProcId, str] = field(default_factory=dict)
    unreadable: List[int] = field(default_factory=list)
    scope_uid: Optional[int] = None

    @property
    def deleted_bytes(self) -> int:
        return sum(log.allocated for log in self.logs)

    def log_for(self, key: procfs.FileKey) -> Optional[LogRecord]:
        for log in self.logs:
            if log.key == key:
                return log
        return None


def _canonical_path(targets: Sequence[str], roots: Sequence[str]) -> str:
    """Pick a stable display path, preferring one inside a log root."""
    cleaned = [t for t in targets if t]
    if not cleaned:
        return ""
    for root in roots:
        prefix = root.rstrip("/") + "/"
        inside = sorted(t for t in cleaned if t == root or t.startswith(prefix))
        if inside:
            return inside[0]
    return sorted(cleaned)[0]


def collect(
    scope_uid: Optional[int] = None,
    log_roots: Sequence[str] = (),
    only_pids: Optional[Sequence[int]] = None,
) -> Scan:
    """Enumerate processes, group deleted open files, build candidates.

    ``only_pids`` restricts the view to a known set of writers: the page then
    reports exactly the files those processes hold, instead of every deleted
    file on the host.
    """
    processes = procfs.list_processes(only_pids=only_pids, uid=scope_uid)

    unreadable: List[int] = []
    if only_pids is None:
        # Warn about processes whose descriptor table we cannot read: their
        # holdings are invisible, so a plan could under-count holders.
        for pid in procfs.list_pids():
            try:
                os.listdir(f"/proc/{pid}/fd")
            except PermissionError:
                unreadable.append(pid)
            except OSError:
                pass

    groups = procfs.deleted_groups(processes)
    by_pid = {p.pid: p for p in processes}
    logs: List[LogRecord] = []
    for key, items in groups.items():
        holders: Dict[procfs.ProcId, List[int]] = {}
        targets: List[str] = []
        blocks = 0
        links = 0
        for item in items:
            proc = by_pid.get(item.pid)
            if proc is None:
                continue
            holders.setdefault(proc.ident, []).append(item.fd)
            targets.append(item.target)
            blocks = max(blocks, item.blocks)
            links = max(links, item.nlink)
        if not holders:
            continue
        for fds in holders.values():
            fds.sort()
        logs.append(
            LogRecord(
                key=key,
                path=_canonical_path(targets, log_roots),
                blocks=blocks,
                holders=holders,
                inode_links=links,
            )
        )
    logs = dedupe_logs(logs)

    candidates: List[Candidate] = []
    holder_ids = set()
    for log in logs:
        holder_ids.update(log.holders)
    for proc in processes:
        if proc.ident not in holder_ids:
            continue
        candidates.append(
            Candidate(
                process=proc.ident,
                display=proc.display,
                comm=proc.comm,
                cmdline=list(proc.cmdline),
                uid=proc.uid,
            )
        )
    hold_map: Dict[procfs.ProcId, set] = {}
    for log in logs:
        for proc_id in log.holders:
            hold_map.setdefault(proc_id, set()).add(log.key)
    for cand in candidates:
        cand.holds = frozenset(hold_map.get(cand.process, set()))
    candidates.sort(key=lambda c: c.process.pid)

    display = {p.ident: p.display for p in processes}
    return Scan(
        processes=processes,
        logs=logs,
        candidates=candidates,
        display=display,
        unreadable=unreadable,
        scope_uid=scope_uid,
    )
