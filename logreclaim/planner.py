"""Planning: choose the fewest processes whose restart would free enough space.

Model
-----
A *deleted log* is a regular file that has been unlinked but is still open by
at least one process.  It is identified by ``(st_dev, st_ino)`` so the same
inode opened through several descriptors or by several processes is counted
once.  Its reclaimable size is the real allocation, ``st_blocks * 512``.

Space is only credited to a candidate set ``S`` when, for a given log, **every**
process currently holding it is in ``S``: as long as one holder survives, the
kernel keeps the inode and all of its blocks alive.  Formally ``S`` releases
log ``L`` iff ``holders(L) ⊆ S``.

That turns the problem into a minimum-cardinality set cover over the deleted
logs: each log is a set of holder processes, and we need the fewest processes
whose union of "fully covered logs" reaches the target.  Among equally small
sets we prefer the one freeing the most bytes.

This module is pure logic (no I/O), which keeps it directly unit-testable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .procfs import FileKey, ProcId

# Exact subset search is cheap for a handful of processes.  Beyond this we
# fall back to a greedy cover so an unexpectedly large process table cannot
# stall the request.
EXACT_SEARCH_LIMIT = 18


@dataclass
class LogRecord:
    """A deleted-but-open log file and the processes holding it."""

    key: FileKey
    path: str
    blocks: int                       # st_blocks, 512-byte sectors
    holders: Dict[ProcId, List[int]]  # process -> fds referencing this inode
    inode_links: int = 0              # surviving hard links, informational

    @property
    def allocated(self) -> int:
        return self.blocks * 512

    @property
    def holder_ids(self) -> FrozenSet[ProcId]:
        return frozenset(self.holders)


@dataclass
class Candidate:
    process: ProcId
    display: str
    comm: str
    cmdline: List[str]
    uid: int
    holds: FrozenSet[FileKey] = frozenset()
    open_paths: Tuple[str, ...] = ()   # configured current log paths

    @property
    def label(self) -> str:
        return f"{self.process.pid} · {self.display}"


@dataclass
class Plan:
    target_bytes: int
    candidates: List[Candidate]
    logs: List[LogRecord]
    selected: List[Candidate] = field(default_factory=list)
    covered: List[LogRecord] = field(default_factory=list)
    freed_bytes: int = 0
    feasible: bool = False
    unreachable: List[LogRecord] = field(default_factory=list)
    released_by: Dict[ProcId, List[LogRecord]] = field(default_factory=dict)
    exclusive_bytes: Dict[ProcId, int] = field(default_factory=dict)
    note: str = ""

    @property
    def selected_ids(self) -> Tuple[ProcId, ...]:
        return tuple(c.process for c in self.selected)

    @property
    def covered_keys(self) -> Tuple[FileKey, ...]:
        return tuple(log.key for log in self.covered)


def dedupe_logs(logs: Iterable[LogRecord]) -> List[LogRecord]:
    """One LogRecord per (dev, ino); merge holders and fd lists."""
    merged: Dict[FileKey, LogRecord] = {}
    for log in logs:
        current = merged.get(log.key)
        if current is None:
            merged[log.key] = LogRecord(
                key=log.key,
                path=log.path,
                blocks=log.blocks,
                holders={p: list(fds) for p, fds in log.holders.items()},
                inode_links=log.inode_links,
            )
            continue
        # Different holders should report the same allocation; max is safe.
        current.blocks = max(current.blocks, log.blocks)
        current.inode_links = max(current.inode_links, log.inode_links)
        if not current.path and log.path:
            current.path = log.path
        for proc_id, fds in log.holders.items():
            slot = current.holders.setdefault(proc_id, [])
            for fd in fds:
                if fd not in slot:
                    slot.append(fd)
            slot.sort()
    return sorted(merged.values(), key=lambda r: (-r.allocated, r.key))


def can_be_freed(log: LogRecord) -> bool:
    """Closing every descriptor is enough only when no other hard link exists."""
    return log.inode_links <= 0


def released_logs(
    logs: Sequence[LogRecord], selected: Iterable[ProcId]
) -> List[LogRecord]:
    """Logs whose complete holder set is contained in ``selected``."""
    chosen = frozenset(selected)
    return [
        log
        for log in logs
        if can_be_freed(log) and set(log.holders).issubset(chosen)
    ]


def freed_bytes_for(
    logs: Sequence[LogRecord], selected: Iterable[ProcId]
) -> int:
    return sum(log.allocated for log in released_logs(logs, selected))


def _build_candidates(
    logs: Sequence[LogRecord],
    candidates: Sequence[Candidate],
) -> List[Candidate]:
    hold_map: Dict[ProcId, set] = {}
    for log in logs:
        for proc_id in log.holders:
            hold_map.setdefault(proc_id, set()).add(log.key)
    out: List[Candidate] = []
    for cand in candidates:
        if not cand.holds:
            cand.holds = frozenset(hold_map.get(cand.process, set()))
        out.append(cand)
    return out


def _greedy_cover(
    logs: Sequence[LogRecord],
    candidate_ids: Sequence[ProcId],
    target_bytes: int,
) -> Optional[FrozenSet[ProcId]]:
    """Greedy fallback: take the log that needs the fewest new processes,
    preferring larger allocations, until the target is met."""
    allowed = set(candidate_ids)
    selected: set = set()
    freed = 0
    remaining = list(logs)
    while remaining and freed < target_bytes:
        best = None
        best_score = None
        for log in remaining:
            missing = set(log.holders) - selected
            if not missing or not missing <= allowed:
                continue
            score = (len(missing), -log.allocated)
            if best_score is None or score < best_score:
                best, best_score = log, score
        if best is None:
            break
        selected |= set(best.holders)
        newly = [log for log in remaining if set(log.holders) <= selected]
        freed += sum(log.allocated for log in newly)
        remaining = [log for log in remaining if log not in newly]
    if freed < target_bytes:
        return None
    return frozenset(selected)


def compute_plan(
    logs: Sequence[LogRecord],
    candidates: Sequence[Candidate],
    target_bytes: int,
) -> Plan:
    """Find the fewest processes that free at least ``target_bytes``."""
    deduped = dedupe_logs(logs)
    cand_by_id = {c.process: c for c in candidates}
    for cand in cand_by_id.values():
        if not cand.holds:
            cand.holds = frozenset(l.key for l in deduped if cand.process in l.holders)
    candidate_ids = sorted(cand_by_id)

    reachable: List[LogRecord] = []
    unreachable: List[LogRecord] = []
    for log in deduped:
        if not can_be_freed(log):
            # Another directory entry still names this inode, so reopening
            # descriptors cannot release its blocks.
            unreachable.append(log)
        elif set(log.holders).issubset(cand_by_id):
            reachable.append(log)
        else:
            unreachable.append(log)

    plan = Plan(
        target_bytes=target_bytes,
        candidates=list(cand_by_id.values()),
        logs=deduped,
        unreachable=unreachable,
    )

    if target_bytes <= 0:
        plan.note = "目标释放量必须大于 0。"
        return plan
    if not reachable:
        if unreachable:
            plan.note = (
                "没有可被现有进程完整释放的已删除日志："
                "部分日志仍有其他链接，或持有者超出本页可控制的进程。"
            )
        else:
            plan.note = "没有可被现有进程完整释放的已删除日志。"
        return plan

    total_possible = sum(log.allocated for log in reachable)
    if total_possible < target_bytes:
        full = frozenset(candidate_ids)
        released = released_logs(reachable, full)
        plan.selected = sorted(
            (cand_by_id[p] for p in full), key=lambda c: c.process.pid
        )
        plan.covered = released
        plan.freed_bytes = sum(log.allocated for log in released)
        plan.note = (
            f"即使让全部 {len(full)} 个候选进程重开日志，最多也只能释放 "
            f"{plan.freed_bytes} 字节，达不到目标。"
        )
        return plan

    count = len(candidate_ids)
    best_subset: Optional[FrozenSet[ProcId]] = None
    best_freed = -1
    if count <= EXACT_SEARCH_LIMIT:
        # Smallest cardinality first, then largest freed amount.
        for size in range(1, count + 1):
            for combo in itertools.combinations(candidate_ids, size):
                subset = frozenset(combo)
                freed = freed_bytes_for(reachable, subset)
                if freed >= target_bytes and freed > best_freed:
                    best_subset, best_freed = subset, freed
            if best_subset is not None:
                break
    else:
        greedy = _greedy_cover(reachable, candidate_ids, target_bytes)
        if greedy is not None:
            best_subset = greedy
            best_freed = freed_bytes_for(reachable, greedy)

    if best_subset is None:
        plan.note = "在全部候选进程中未找到能达标的组合。"
        return plan

    plan.selected = sorted(
        (cand_by_id[p] for p in best_subset), key=lambda c: c.process.pid
    )
    plan.covered = released_logs(reachable, best_subset)
    plan.freed_bytes = sum(log.allocated for log in plan.covered)
    plan.feasible = True
    for log in plan.covered:
        holders = list(log.holders)
        for proc_id in holders:
            plan.released_by.setdefault(proc_id, []).append(log)
        if len(holders) == 1:
            # Only this process holds the file, so the whole file is its doing.
            proc_id = holders[0]
            plan.exclusive_bytes[proc_id] = (
                plan.exclusive_bytes.get(proc_id, 0) + log.allocated
            )
    plan.note = (
        f"最少需要 {len(plan.selected)} 个进程，当前可释放 "
        f"{plan.freed_bytes} 字节。"
    )
    return plan
