#!/usr/bin/env python3
"""Seed a demo generation of deleted logs, for use outside Docker.

    python -m demo.seed --log-root /tmp/logs --state-dir /tmp/state

Starts the demo writers, gives them a moment to write, then rotates and
unlinks their logs so the page immediately has something to reclaim.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import supervise  # noqa: E402
from logreclaim.config import Config  # noqa: E402
from logreclaim.controller import Controller  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", default=os.environ.get("LOG_ROOT", "/data/logs"))
    parser.add_argument("--state-dir", default=os.environ.get("STATE_DIR", "/data/state"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--fill-seconds", type=float, default=2.0)
    parser.add_argument("--chunk", type=int, default=int(os.environ.get("WRITE_CHUNK", "8192")))
    parser.add_argument("--interval", type=float, default=float(os.environ.get("WRITE_INTERVAL", "0.25")))
    parser.add_argument("--max-mb", type=float, default=float(os.environ.get("WRITE_MAX_MB", "64")))
    args = parser.parse_args(argv)

    if supervise.running(args.state_dir):
        print("writers already running; seeding a new generation")
    else:
        supervise.main([
            "--log-root", args.log_root,
            "--state-dir", args.state_dir,
            "--count", str(args.count),
            "--chunk", str(args.chunk),
            "--interval", str(args.interval),
            "--max-mb", str(args.max_mb),
        ])
    time.sleep(args.fill_seconds)

    os.environ["LOG_ROOT"] = args.log_root
    os.environ["STATE_DIR"] = args.state_dir
    controller = Controller(Config.from_env())
    moved = controller.demo_rotate()
    deleted = controller.demo_delete_old()
    total = sum(size for _, size in deleted)
    print(f"rotated {len(moved)} files, deleted {len(deleted)} old logs "
          f"({total} bytes still held by writers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
