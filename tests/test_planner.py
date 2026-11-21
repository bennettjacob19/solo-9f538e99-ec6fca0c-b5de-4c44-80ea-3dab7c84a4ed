"""Unit tests for the pure planning logic (no /proc, no Flask)."""

from __future__ import annotations

from logreclaim.planner import (
    Candidate,
    LogRecord,
    compute_plan,
    dedupe_logs,
    freed_bytes_for,
    released_logs,
)
from logreclaim.procfs import FileKey, ProcId


def proc(pid: int, start: int = 1) -> ProcId:
    return ProcId(pid, start)


def cand(*pids: ProcId) -> list:
    return [Candidate(p, f"p{p.pid}", f"p{p.pid}", [], 0) for p in pids]


def log(ino: int, blocks: int, *pids: ProcId) -> LogRecord:
    return LogRecord(
        key=FileKey(1, ino),
        path=f"/logs/{ino}.log",
        blocks=blocks,
        holders={p: [3] for p in pids},
    )


A, B, C, D = proc(10), proc(20), proc(30), proc(40)


def test_dedupe_collapses_same_device_and_inode():
    shared = log(7, 512, A, B)
    same = LogRecord(FileKey(1, 7), "/other/path", 512, {B: [9], C: [4]})
    merged = dedupe_logs([shared, same])
    assert len(merged) == 1
    assert set(merged[0].holders) == {A, B, C}
    assert merged[0].holders[B] == [3, 9]


def test_different_devices_with_same_inode_are_distinct():
    a = LogRecord(FileKey(1, 42), "/a", 8, {A: [3]})
    b = LogRecord(FileKey(2, 42), "/b", 8, {A: [4]})
    assert len(dedupe_logs([a, b])) == 2
    assert freed_bytes_for([a, b], [A]) == 16 * 512


def test_partial_holders_do_not_count():
    # A shared file needs every holder in the selected set.
    shared = log(1, 100, A, B)
    solo = log(2, 60, C)
    logs, cs = [shared, solo], cand(A, B, C)
    assert freed_bytes_for(logs, [A]) == 0
    assert freed_bytes_for(logs, [A, B]) == 100 * 512
    assert [l.key.ino for l in released_logs(logs, [A, B])] == [1]


def test_minimum_cover_prefers_the_single_big_holder():
    # One process holds 3 files worth 300 blocks total; another holds one of
    # 200 blocks.  Target 250 must pick the single-process 300-block set.
    logs = [log(1, 120, A), log(2, 120, A), log(3, 60, A), log(4, 200, B)]
    plan = compute_plan(logs, cand(A, B), 250 * 512)
    assert plan.feasible
    assert [c.process for c in plan.selected] == [A]
    assert plan.freed_bytes == 300 * 512


def test_minimum_cover_requires_shared_holders_together():
    logs = [log(1, 500, A, B), log(2, 50, C)]
    plan = compute_plan(logs, cand(A, B, C), 400 * 512)
    assert [c.process for c in plan.selected] == [A, B]
    assert plan.freed_bytes == 500 * 512


def test_plan_selects_fewest_not_most_bytes():
    # 1 process can free 100; 2 processes free 400.  Target 90 -> one process.
    logs = [log(1, 100, A), log(2, 300, B, C)]
    plan = compute_plan(logs, cand(A, B, C), 90 * 512)
    assert len(plan.selected) == 1
    assert plan.selected[0].process == A

    # 400 blocks requires all three; the two-process set only frees 300.
    plan = compute_plan(logs, cand(A, B, C), 350 * 512)
    assert len(plan.selected) == 3
    assert plan.freed_bytes == 400 * 512


def test_tie_breaks_on_larger_freed_amount():
    logs = [log(1, 100, A), log(2, 200, B)]
    plan = compute_plan(logs, cand(A, B), 50 * 512)
    assert [c.process for c in plan.selected] == [B]


def test_infeasible_reports_best_effort_and_note():
    logs = [log(1, 100, A), log(2, 100, B)]
    plan = compute_plan(logs, cand(A, B), 10_000 * 512)
    assert not plan.feasible
    assert plan.freed_bytes == 200 * 512
    assert "达不到目标" in plan.note


def test_unreachable_when_holder_outside_candidates():
    logs = [log(1, 100, A, D)]  # D has no Candidate -> A alone is not enough
    plan = compute_plan(logs, cand(A), 1)
    assert not plan.feasible
    assert plan.unreachable and plan.unreachable[0].key.ino == 1


def test_zero_target_is_rejected():
    plan = compute_plan([log(1, 10, A)], cand(A), 0)
    assert not plan.feasible
    assert "大于 0" in plan.note


def test_blocks_are_counted_not_apparent_size():
    # A sparse-looking 1 GiB file allocated only 8 blocks frees 4 KiB.
    big = LogRecord(FileKey(1, 9), "/sparse.log", 8, {A: [3]})
    plan = compute_plan([big], cand(A), 1)
    assert plan.freed_bytes == 8 * 512


def test_exact_search_handles_many_holders():
    logs = [log(i, 10, proc(100 + i)) for i in range(1, 15)]
    processes = [proc(100 + i) for i in range(1, 15)]
    plan = compute_plan(logs, cand(*processes), 25 * 512)
    assert plan.feasible
    assert len(plan.selected) == 3
    assert plan.freed_bytes >= 25 * 512
