#!/usr/bin/env python3
"""Supervise the up-to-8 demo writer processes.

Simulates a real fleet of log writers:

* 3 processes share one ``shared.log`` inode (``--mode shared``), so that file
  is only reclaimable when all three reopen.
* 2 processes share a second inode, ``shared-alt.log``, across *different*
  multi-file writers.
* 2 processes each hold three of their own log files (``--mode multi``), so a
  single process accounts for several deleted logs.
* 3 further processes each write a single log.

Writers are started detached from the web process; the web app only learns
about them through the registry files they write under ``$STATE_DIR/agents``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import List

WRITER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "writer.py")

DEFAULT_FLEET = [
    # name, mode, index  (at most 8, matching MAX_WRITERS)
    ("shared-a", "shared", 0),
    ("shared-b", "shared", 0),
    ("shared-c", "shared", 0),
    ("multi-a", "multi,shared-alt", 1),
    ("multi-b", "multi,shared-alt", 2),
    ("solo-a", "single", 3),
    ("solo-b", "single", 4),
    ("solo-c", "single", 5),
]


def agent_state_dir(state_dir: str) -> str:
    return os.path.join(state_dir, "agents")


def _starttime(pid: int) -> int:
    raw = open(f"/proc/{pid}/stat", "rb").read().decode("utf-8", "replace")
    return int(raw[raw.rfind(")") + 2 :].split()[19])


def _still_running(data: dict) -> bool:
    """PID liveness is not enough: a reused PID is a different process."""
    try:
        pid = int(data["pid"])
        starttime = int(data["starttime"])
        socket_path = str(data["socket"])
    except (KeyError, ValueError, TypeError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        if _starttime(pid) != starttime:
            return False
    except OSError:
        return False
    return os.path.exists(socket_path)


def running(state_dir: str) -> List[dict]:
    directory = agent_state_dir(state_dir)
    out = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if _still_running(data):
            out.append(data)
    return out


def clear_stale(state_dir: str, fleet) -> None:
    """Remove registry entries left behind by a previous container/run."""
    directory = agent_state_dir(state_dir)
    names = {name for name, _mode, _index in fleet}
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return
    for entry in entries:
        base, dot, ext = entry.partition(".")
        if base not in names or ext not in ("json", "sock"):
            continue
        path = os.path.join(directory, entry)
        try:
            if ext == "json":
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if _still_running(data):
                    continue
            os.unlink(path)
        except (OSError, ValueError):
            continue


def spawn(name: str, mode: str, index: int, args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        sys.executable, "-u", WRITER,
        "--name", name,
        "--index", str(index),
        "--mode", mode,
        "--log-root", args.log_root,
        "--control", args.state_dir,
        "--chunk", str(args.chunk),
        "--interval", str(args.interval),
        "--max-mb", str(args.max_mb),
    ]
    log_path = os.path.join(args.state_dir, "writer-logs", f"{name}.out")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    stdout = open(log_path, "ab")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stdout,
        start_new_session=True,
        cwd="/",
    )
    stdout.close()
    return proc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="start demo writers")
    parser.add_argument("--log-root", default=os.environ.get("LOG_ROOT", "/data/logs"))
    parser.add_argument("--state-dir", default=os.environ.get("STATE_DIR", "/data/state"))
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--max-mb", type=float, default=64.0)
    parser.add_argument("--count", type=int, default=len(DEFAULT_FLEET))
    parser.add_argument("--wait", type=float, default=8.0,
                        help="seconds to wait for registration before exiting")
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.log_root, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)
    existing = running(args.state_dir)
    if existing:
        print(f"{len(existing)} writers already registered; nothing to do.")
        return 0

    count = max(0, min(args.count, 8))
    fleet = DEFAULT_FLEET[:count]
    clear_stale(args.state_dir, fleet)
    children = [spawn(name, mode, index, args) for name, mode, index in fleet]
    deadline = time.time() + args.wait
    while time.time() < deadline:
        if len(running(args.state_dir)) >= len(fleet):
            break
        time.sleep(0.2)
    ready = running(args.state_dir)
    print(f"started {len(children)} writers, {len(ready)} registered:")
    for item in ready:
        print(f"  pid {item['pid']:>6}  {item['name']:<10} {item.get('modes')}")
    if args.foreground:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            for child in children:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except OSError:
                    pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
