"""End-to-end tests against real Linux /proc, real unlinked files and processes.

These tests do not stub anything: they fork writer processes, delete the files
they hold, read /proc to build a plan, and ask the processes (over their Unix
socket control channel) to reopen.  They are skipped on non-Linux hosts.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.path.isdir("/proc"),
    reason="requires Linux /proc",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRITER = os.path.join(ROOT, "demo", "writer.py")


class Fleet:
    """A tiny supervised fleet living in a temp dir."""

    def __init__(self, tmp: str):
        self.tmp = tmp
        self.log_root = os.path.join(tmp, "logs")
        self.state = os.path.join(tmp, "state")
        os.makedirs(self.log_root, exist_ok=True)
        os.makedirs(self.state, exist_ok=True)
        self.procs = []

    def start(self, name: str, mode: str, extra=()):
        cmd = [
            sys.executable, "-u", WRITER,
            "--name", name,
            "--mode", mode,
            "--log-root", self.log_root,
            "--control", self.state,
            "--chunk", "4096",
            "--interval", "0.05",
            *extra,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.procs.append(proc)
        return proc

    def wait_registered(self, count: int, timeout: float = 10.0):
        directory = os.path.join(self.state, "agents")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ready = [n for n in os.listdir(directory) if n.endswith(".json")]
            except OSError:
                ready = []
            if len(ready) >= count:
                return
            time.sleep(0.05)
        raise AssertionError(f"only {len(ready)} writers registered")

    def stop(self):
        for proc in self.procs:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
        for proc in self.procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass


@pytest.fixture()
def fleet(tmp_path):
    f = Fleet(str(tmp_path))
    yield f
    f.stop()


def send_reopen(fleet, socket_path, paths):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(3)
        sock.connect(socket_path)
        sock.sendall(json.dumps({"op": "reopen", "paths": paths}).encode() + b"\n")
        raw = b""
        while not raw.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            raw += chunk
    return json.loads(raw)


def make_controller(fleet):
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
    )
    return Controller(cfg)


def wait_for_blocks(controller, key, minimum: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = controller.scan()
        for log in view.logs:
            if log.key == key and log.blocks >= minimum:
                return log
        time.sleep(0.05)
    raise AssertionError(f"inode {key} never reached {minimum} blocks")


def test_shared_holders_are_needed_together(fleet):
    fleet.start("s1", "shared")
    fleet.start("s2", "shared")
    fleet.wait_registered(2)
    controller = make_controller(fleet)
    shared = os.path.join(fleet.log_root, "shared.log")
    assert os.path.exists(shared)
    time.sleep(0.4)

    controller.demo_rotate()
    controller.demo_delete_old()
    view = controller.scan()
    totals = [log for log in view.logs if log.path.endswith("shared.log.rot")
              or "shared.log." in log.path]
    assert len(totals) == 1, view.logs
    log = totals[0]
    assert len(log.holders) == 2

    devices = {os.stat(fleet.log_root).st_dev}
    assert log.key.dev in devices  # same filesystem

    plan_one = controller.make_plan(view, log.allocated + 1)
    assert not plan_one.feasible  # impossible: only 2 processes, file needs both
    # but a target within the file's size is reachable with both
    target = max(1, log.allocated // 2)
    plan = controller.make_plan(view, target)
    assert plan.feasible
    assert len(plan.selected) == 2
    assert plan.freed_bytes == log.allocated


def test_execute_releases_original_inode(fleet):
    fleet.start("a", "shared")
    fleet.start("b", "shared")
    fleet.wait_registered(2)
    controller = make_controller(fleet)
    time.sleep(0.4)
    controller.demo_rotate()
    controller.demo_delete_old()

    view = controller.scan()
    assert len(view.logs) == 1
    log = view.logs[0]
    target = max(1, log.allocated // 2)
    plan = controller.make_plan(view, target)
    assert plan.feasible and len(plan.selected) == 2
    snap = controller.snapshot(view, plan, target / (1024 * 1024))
    report = controller.execute(snap)

    assert report.executed, report.reasons
    assert [r.ok for r in report.results] == [True, True]
    assert [r.key for r in report.released] == [log.key]
    assert report.freed_bytes >= target
    assert report.remaining == []
    # Nothing deleted is held any more.
    assert controller.scan().logs == []
    # Writers created fresh files and kept appending.
    assert os.path.getsize(os.path.join(fleet.log_root, "shared.log")) > 0


def test_stale_plan_is_refused_when_the_old_file_was_already_reopened(fleet):
    fleet.start("x", "single")
    fleet.wait_registered(1)
    controller = make_controller(fleet)
    time.sleep(0.4)
    controller.demo_rotate()
    controller.demo_delete_old()
    view = controller.scan()
    assert len(view.logs) == 1
    log = view.logs[0]
    plan = controller.make_plan(view, max(1, log.allocated // 2))
    snap = controller.snapshot(view, plan, plan.target_bytes / (1024 * 1024))

    # Simulate an out-of-band reopen: the writer lets go of the deleted inode
    # and starts a fresh current file.  The saved plan is now describing a
    # file that no longer exists.
    agent = view.agent_for(plan.selected[0].process)
    reply = send_reopen(fleet, agent.info.socket, agent.info.paths)
    assert all(item["reopened"] for item in reply["results"])
    time.sleep(0.2)
    assert controller.scan().logs == []

    report = controller.execute(snap)
    assert not report.executed
    assert any("不再被打开" in reason for reason in report.reasons)
    # Nothing was reopened a second time: the current file was untouched.
    assert os.path.exists(os.path.join(fleet.log_root, "x.log"))


def test_plan_is_rejected_when_no_longer_minimal(fleet):
    fleet.start("a", "single")
    fleet.start("b", "single")
    fleet.wait_registered(2)
    controller = make_controller(fleet)
    time.sleep(0.6)
    controller.demo_rotate()
    controller.demo_delete_old()
    view = controller.scan()
    assert len(view.logs) == 2
    # A target above either file alone forces both writers into the plan.
    sizes = sorted(log.allocated for log in view.logs)
    target = sizes[-1] + 1
    plan = controller.make_plan(view, target)
    assert len(plan.selected) == 2
    snap = controller.snapshot(view, plan, target / (1024 * 1024))

    # A brand new writer appears whose single deleted log alone meets the
    # target.  The old two-process list still covers its own logs, so the only
    # problem is that it is no longer the smallest list.
    fleet.start("c", "single")
    fleet.wait_registered(3)
    deadline = time.time() + 10
    c_path = os.path.join(fleet.log_root, "c.log")
    while time.time() < deadline:
        if os.path.exists(c_path) and os.path.getsize(c_path) > target:
            break
        time.sleep(0.05)
    os.rename(c_path, c_path + ".rot")
    os.unlink(c_path + ".rot")
    time.sleep(0.1)

    report = controller.execute(snap)
    assert not report.executed
    assert "不再最少" in " ".join(report.reasons)
    # The old logs were not touched.
    held = {item.key for item in controller.scan().logs}
    assert {item.key for item in snap.covered} <= held


def test_stale_plan_refused_when_holder_exits(fleet):
    p1 = fleet.start("h1", "shared")
    fleet.start("h2", "shared")
    fleet.wait_registered(2)
    controller = make_controller(fleet)
    time.sleep(0.4)
    controller.demo_rotate()
    controller.demo_delete_old()
    view = controller.scan()
    log = view.logs[0]
    plan = controller.make_plan(view, max(1, log.allocated // 2))
    snap = controller.snapshot(view, plan, plan.target_bytes / (1024 * 1024))

    os.killpg(p1.pid, signal.SIGTERM)
    p1.wait(timeout=5)
    time.sleep(0.2)

    report = controller.execute(snap)
    assert not report.executed
    assert any("持有者" in reason or "退出" in reason for reason in report.reasons)


def test_reopen_failure_keeps_original_handle(fleet):
    """A writer whose current path cannot be recreated keeps writing on the
    original descriptor."""
    fleet.start("locked", "single")
    fleet.wait_registered(1)
    controller = make_controller(fleet)
    time.sleep(0.4)
    controller.demo_rotate()
    controller.demo_delete_old()
    view = controller.scan()
    log = view.logs[0]
    plan = controller.make_plan(view, max(1, log.allocated // 2))
    snap = controller.snapshot(view, plan, plan.target_bytes / (1024 * 1024))

    # Make the parent directory read-only so O_CREAT fails for the writer.
    os.chmod(fleet.log_root, 0o500)
    try:
        report = controller.execute(snap)
    finally:
        os.chmod(fleet.log_root, 0o755)

    assert report.executed
    assert report.results and not report.results[0].ok
    assert "沿用原句柄" in report.results[0].detail
    # Original inode survives and is still attributed to the writer.
    assert [item.key for item in report.remaining] == [log.key]
    assert report.remaining_holders[log.key]


def test_web_flow_plan_confirm(fleet):
    fleet.start("w1", "shared")
    fleet.start("w2", "shared")
    fleet.wait_registered(2)
    from logreclaim.app import create_app

    controller = make_controller(fleet)
    app = create_app(controller.config)
    app.extensions["logreclaim"] = controller
    client = app.test_client()

    time.sleep(0.4)
    controller.demo_rotate()
    controller.demo_delete_old()
    view = controller.scan()
    target = max(1, view.logs[0].allocated // 2)
    resp = client.post("/plan", data={"target_mib": str(target / (1024 * 1024))})
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    import re

    token = re.search(r"/confirm/([A-Za-z0-9_-]+)", body)
    assert token
    resp = client.post(f"/confirm/{token.group(1)}")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert "已释放" in text or "确认释放" in text
    assert "仍在占用" in text

    # Replaying the same token must not execute twice.
    resp = client.post(f"/confirm/{token.group(1)}")
    assert resp.status_code == 302
