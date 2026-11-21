"""Integration tests for the two-phase SCAN_SCOPE=host experience.

These tests exercise real /proc and real writer processes:

* the home page first shows the controlled-writer screen while a full-host
  scan runs in the background, then swaps to the complete table;
* a failing background scan leaves the first screen in place and allows a
  manual retry;
* rotation / deletion / confirmed reclamation bump a table epoch, so only a
  scan started afterwards can publish, and a stuck or late old scan can never
  overwrite the newer table;
* plan calculation and confirmation always run real-time full-host scans.
"""

from __future__ import annotations

import threading
import time

import pytest

from test_integration import Fleet, fleet  # noqa: F401  (reuse the real fleet)

pytestmark = pytest.mark.skipif(
    not __import__("sys").platform.startswith("linux")
    or not __import__("os").path.isdir("/proc"),
    reason="requires Linux /proc",
)


def make_host_controller(fleet):
    from logreclaim.config import Config
    from logreclaim.controller import Controller

    cfg = Config(
        log_root=fleet.log_root,
        state_dir=fleet.state,
        secret_key="test",
        host="127.0.0.1",
        port=0,
        demo_enabled=True,
        max_writers=8,
        plan_ttl_seconds=60,
        write_chunk=4096,
        write_interval=0.05,
        scan_scope="host",
    )
    return Controller(cfg)


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.05):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"condition not met; last value={last!r}")


def host_scan_threads():
    return [
        t for t in threading.enumerate()
        if t.name.startswith("host-scan-") and t.is_alive()
    ]


def test_first_screen_is_agents_only_until_background_scan_completes(fleet):
    fleet.start("w1", "shared")
    fleet.start("w2", "shared")
    fleet.wait_registered(2)
    controller = make_host_controller(fleet)
    time.sleep(0.4)
    controller.demo_rotate()
    controller.demo_delete_old()

    gate = threading.Event()
    controller.host_scan_gate = gate

    first = controller.index_view()
    status = first.host_status
    assert status.host_scope is True
    assert status.is_running
    # First screen: exactly the registered writer processes, their deleted
    # logs, real allocation and known holders.
    writer_pids = {a.info.pid for a in first.agents_live}
    assert {p.pid for p in first.scan.processes} == writer_pids
    assert first.logs, "the writers' held deleted logs must be on the first screen"
    assert all(log.blocks > 0 for log in first.logs)
    assert all(log.holders for log in first.logs)

    # Let the background full-host scan finish and wait for the table swap.
    gate.set()
    wait_until(
        lambda: controller.host_scan_status().state == "ready"
    )
    full = controller.index_view()
    assert full.host_status.state == "ready"
    assert full.host_status.table_is_host_wide
    # The host-wide pass sees every visible process, not just the writers.
    full_pids = {p.pid for p in full.scan.processes}
    assert writer_pids <= full_pids
    assert len(full_pids) >= len(writer_pids)
    # The same deleted logs are still reported, now with complete holder data.
    assert {log.key for log in first.logs} <= {log.key for log in full.logs}


def test_agents_scope_has_no_background_machinery(fleet):
    from logreclaim.config import Config
    from logreclaim.controller import Controller, HOST_SCAN_DISABLED

    fleet.start("w1", "single")
    fleet.wait_registered(1)
    cfg = Config(
        log_root=fleet.log_root,
        state_dir=fleet.state,
        secret_key="test",
        host="127.0.0.1",
        port=0,
        demo_enabled=True,
        max_writers=8,
        plan_ttl_seconds=60,
        write_chunk=4096,
        write_interval=0.05,
        scan_scope="agents",
    )
    controller = Controller(cfg)
    view = controller.index_view()
    assert view.host_status.state == HOST_SCAN_DISABLED
    assert host_scan_threads() == []


def test_reclamation_during_slow_scan_uses_fresh_scan_and_stale_result_is_discarded(fleet):
    fleet.start("a", "shared")
    fleet.start("b", "shared")
    fleet.wait_registered(2)
    controller = make_host_controller(fleet)
    time.sleep(0.4)
    controller.demo_rotate()
    controller.demo_delete_old()

    # First background full-host scan gets stuck before it can read /proc.
    gate_one = threading.Event()
    controller.host_scan_gate = gate_one
    first = controller.index_view()
    assert first.host_status.is_running
    epoch_one = first.host_status.table_epoch
    wait_until(lambda: len(host_scan_threads()) == 1)

    # Plan + confirmed reclamation run on real-time synchronous full scans,
    # unaffected by the stuck background worker.
    view = controller.scan()
    assert view.host_status.state == "live"
    log = view.logs[0]
    target = max(1, log.allocated // 2)
    plan = controller.make_plan(view, target)
    assert plan.feasible
    snap = controller.snapshot(view, plan, target / (1024 * 1024))
    report = controller.execute(snap)
    assert report.executed, report.reasons

    # The mutation barrier advanced: the stuck scan belongs to an old epoch.
    status = controller.host_scan_status()
    epoch_two = status.table_epoch
    assert epoch_two == epoch_one + 1
    assert status.state == "idle"

    # Capture the writers' reclaimed inodes before they are released.
    reclaimed_keys = {item.key for item in report.released}
    assert reclaimed_keys

    gate_two = threading.Event()
    controller.host_scan_gate = gate_two
    again = controller.index_view()
    assert again.host_status.is_running
    assert again.host_status.table_epoch == epoch_two
    wait_until(lambda: len(host_scan_threads()) == 2)

    gate_two.set()
    wait_until(
        lambda: (
            controller.host_scan_status().state == "ready"
            and controller.host_scan_status().table_epoch == epoch_two
        )
    )

    # Now release the ancient, stuck worker; its late result must be dropped.
    gate_one.set()
    time.sleep(0.4)
    status = controller.host_scan_status()
    assert status.state == "ready"
    assert status.table_epoch == epoch_two

    final = controller.index_view()
    assert final.host_status.state == "ready"
    assert final.host_status.table_epoch == epoch_two
    # The reclamation outcome survives: the stale scan cannot resurrect the
    # writers' old inodes.  (A full-host scan may legitimately report deleted
    # files held by unrelated processes, e.g. the test runner itself.)
    final_keys = {log.key for log in final.logs}
    assert reclaimed_keys.isdisjoint(final_keys)


def test_failed_scan_keeps_first_screen_and_retry_succeeds(fleet):
    fleet.start("w1", "single")
    fleet.wait_registered(1)
    controller = make_host_controller(fleet)
    time.sleep(0.4)
    controller.demo_rotate()
    controller.demo_delete_old()

    gate = threading.Event()
    controller.host_scan_gate = gate

    original_build = controller._build_view
    failures = {"background_calls": 0}

    def flaky_build(only_pids=None):
        if threading.current_thread().name.startswith("host-scan-"):
            failures["background_calls"] += 1
            if failures["background_calls"] == 1:
                raise RuntimeError("simulated /proc meltdown")
        return original_build(only_pids=only_pids)

    controller._build_view = flaky_build
    try:
        first = controller.index_view()
        assert first.host_status.is_running
        writer_pids = {p.pid for p in first.scan.processes}
        assert first.logs

        gate.set()
        wait_until(lambda: controller.host_scan_status().is_failed)
        failed = controller.host_scan_status()
        assert "simulated /proc meltdown" in failed.error
        failed_attempt = failed.attempt

        # The first screen stays in place and the home page does not restart
        # the scan on its own while the state is "failed".
        retained = controller.index_view()
        assert retained.host_status.is_failed
        assert retained.host_status.attempt == failed_attempt
        assert {p.pid for p in retained.scan.processes} == writer_pids
        assert retained.logs
    finally:
        controller._build_view = original_build
        controller.host_scan_gate = None

    # Explicit retry starts a fresh background scan that succeeds.
    controller.retry_host_scan()
    wait_until(lambda: controller.host_scan_status().state == "ready")
    recovered = controller.index_view()
    assert recovered.host_status.state == "ready"
    assert recovered.host_status.table_is_host_wide


def test_web_flow_banner_status_retry_and_live_plan(fleet):
    fleet.start("w1", "shared")
    fleet.start("w2", "shared")
    fleet.wait_registered(2)
    from logreclaim.app import create_app

    controller = make_host_controller(fleet)
    gate = threading.Event()
    controller.host_scan_gate = gate

    app = create_app(controller.config)
    app.extensions["logreclaim"] = controller
    client = app.test_client()

    time.sleep(0.4)
    controller.demo_rotate()
    controller.demo_delete_old()

    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "全机核对尚未完成" in body
    assert "受控首屏" in body

    status = client.get("/scan/status").get_json()
    assert status["running"] is True
    assert status["failed"] is False
    assert status["host_wide"] is False

    resp = client.post("/scan/retry")
    assert resp.status_code == 302

    # Planning never consumes the background table: it scans the whole host
    # in real time, even while a background scan is still stuck.
    resp = client.post("/plan", data={"target_mib": "0.01"})
    assert resp.status_code == 200
    assert "实时全机扫描" in resp.get_data(as_text=True)

    gate.set()
    wait_until(lambda: controller.host_scan_status().state == "ready")
    resp = client.get("/")
    body = resp.get_data(as_text=True)
    assert "全机核对已完成" in body
    assert "全机结果" in body
