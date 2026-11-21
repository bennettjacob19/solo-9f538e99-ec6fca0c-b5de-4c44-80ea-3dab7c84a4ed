#!/usr/bin/env python3
"""A demo log writer that behaves like a real long-running service.

It appends to one or more log files forever, and it keeps its original file
descriptor when the file is unlinked underneath it — that is exactly the
situation that leaks disk space on Linux.

Modes (comma separated in ``--mode``):

* ``shared``  — the same inode is opened by several writers, so the file is
  only reclaimable once *all* of them reopen their logs (or exit).
* ``multi``   — one process holds several deleted logs at once.

Control
-------
With ``--control <dir>`` the writer registers a small JSON file and listens on
a Unix socket for these operations:

* ``{"op": "reopen", "paths": [...]}`` — for each requested path, open a fresh
  descriptor (creating a new file if the old one was unlinked) and continue
  writing on it.  Descriptors that fail to reopen are left untouched, so the
  process keeps writing through its original handle.
* ``{"op": "status"}`` — report pid, start time and every descriptor.
* ``{"op": "shutdown"}`` — flush and exit.

Without ``--control`` it simply writes forever (e.g. for a plain container).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

STOP = threading.Event()


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    sys.stderr.write(f"[writer {os.getpid()} {stamp}] {message}\n")
    sys.stderr.flush()


def proc_starttime(pid: int) -> int:
    raw = open(f"/proc/{pid}/stat", "rb").read().decode("utf-8", "replace")
    close = raw.rfind(")")
    fields = raw[close + 2 :].split()
    return int(fields[19])


@dataclass
class Target:
    path: str
    fh: Optional[object] = None
    fd: int = -1
    generation: int = 0
    last_error: str = ""
    written: int = 0

    def open(self, append: bool = True) -> None:
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT
        flags |= os.O_APPEND if append else os.O_TRUNC
        fd = os.open(self.path, flags, 0o644)
        old = self.fh
        self.fh = os.fdopen(fd, "ab", buffering=0)
        self.fd = fd
        self.generation += 1
        self.last_error = ""
        self.written = 0
        if old is not None:
            try:
                old.close()
            except OSError:
                pass

    def write(self, payload: bytes) -> None:
        if self.fh is None:
            return
        try:
            self.fh.write(payload)
        except (OSError, ValueError) as exc:
            self.last_error = getattr(exc, "strerror", None) or str(exc)

    def close(self) -> None:
        if self.fh is not None:
            try:
                self.fh.close()
            except OSError:
                pass
            self.fh = None
            self.fd = -1


class Writer:
    def __init__(self, name: str, args: argparse.Namespace):
        self.name = name
        self.args = args
        self.mode = {m.strip() for m in args.mode.split(",") if m.strip()}
        self.handles: List[Target] = []
        self.lock = threading.Lock()
        self.seq = 0
        self.control: Optional[socket.socket] = None
        self.control_thread: Optional[threading.Thread] = None
        self.registered: Optional[str] = None

    # -- handle management -------------------------------------------------
    def compose_targets(self) -> List[Target]:
        targets: List[Target] = []
        if "shared" in self.mode:
            # Several processes point at this one inode; every one of them
            # must reopen before the old inode can be freed.
            targets.append(Target(os.path.join(self.args.log_root, "shared.log")))
        if "shared-alt" in self.mode:
            targets.append(Target(os.path.join(self.args.log_root, "shared-alt.log")))
        if "multi" in self.mode:
            for slot in (1, 2, 3):
                path = os.path.join(self.args.log_root, f"{self.name}-{slot}.log")
                targets.append(Target(path, generation=slot))
        if not targets:
            targets.append(Target(os.path.join(self.args.log_root, f"{self.name}.log")))
        return targets

    def open_all(self) -> None:
        for target in self.handles:
            target.close()
        self.handles = self.compose_targets()
        for target in self.handles:
            target.open()

    # -- control socket ----------------------------------------------------
    def register(self) -> None:
        if not self.args.control:
            return
        directory = os.path.join(self.args.control, "agents")
        os.makedirs(directory, exist_ok=True)
        socket_path = os.path.join(directory, f"{self.name}.sock")
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(socket_path)
        os.chmod(socket_path, 0o600)
        sock.listen(8)
        sock.settimeout(0.5)
        self.control = sock
        self.registered = socket_path
        payload = {
            "pid": os.getpid(),
            "starttime": proc_starttime(os.getpid()),
            "name": self.name,
            "socket": socket_path,
            "paths": [t.path for t in self.handles],
            "modes": sorted(self.mode),
            "started_at": time.time(),
        }
        meta_path = os.path.join(directory, f"{self.name}.json")
        tmp = meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, meta_path)
        self.control_thread = threading.Thread(
            target=self._serve, name=f"{self.name}-control", daemon=True
        )
        self.control_thread.start()

    def _serve(self) -> None:
        assert self.control is not None
        while not STOP.is_set():
            try:
                conn, _ = self.control.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(2.0)
                try:
                    raw = b""
                    while not raw.endswith(b"\n") and len(raw) < 65536:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        raw += chunk
                    request = json.loads(raw.decode("utf-8") or "{}")
                    response = self.handle(request)
                except (ValueError, OSError) as exc:
                    response = {"ok": False, "error": str(exc)}
                try:
                    conn.sendall(json.dumps(response).encode("utf-8") + b"\n")
                except OSError:
                    pass

    def handle(self, request: Dict) -> Dict:
        op = request.get("op")
        if op == "status":
            return self.status()
        if op == "reopen":
            return self.reopen(request.get("paths") or [])
        if op == "shutdown":
            STOP.set()
            return {"ok": True, "pid": os.getpid()}
        return {"ok": False, "error": f"unknown op {op!r}"}

    def status(self) -> Dict:
        handles = []
        for target in self.handles:
            fd = target.fd
            entry = {
                "path": target.path,
                "fd": fd,
                "generation": target.generation,
                "last_error": target.last_error,
            }
            try:
                st = os.stat(f"/proc/self/fd/{fd}")
                entry.update(
                    {"dev": st.st_dev, "ino": st.st_ino, "blocks": st.st_blocks,
                     "size": st.st_size}
                )
            except OSError as exc:
                entry["stat_error"] = exc.strerror
            handles.append(entry)
        return {
            "ok": True,
            "pid": os.getpid(),
            "starttime": proc_starttime(os.getpid()),
            "name": self.name,
            "mode": sorted(self.mode),
            "handles": handles,
        }

    def reopen(self, paths: List[str]) -> Dict:
        wanted = set(paths)
        results = []
        with self.lock:
            for target in self.handles:
                if wanted and target.path not in wanted:
                    results.append(
                        {"path": target.path, "reopened": False,
                         "reason": "本次未选中"}
                    )
                    continue
                old_fd = target.fd
                try:
                    old_st = os.stat(f"/proc/self/fd/{old_fd}")
                    old_key = (old_st.st_dev, old_st.st_ino)
                except OSError:
                    old_key = None
                try:
                    target.open(append=True)
                except OSError as exc:
                    target.last_error = exc.strerror or str(exc)
                    results.append(
                        {"path": target.path, "reopened": False,
                         "reason": f"重开失败：{target.last_error}"}
                    )
                    continue
                try:
                    new_st = os.stat(f"/proc/self/fd/{target.fd}")
                    new_key = (new_st.st_dev, new_st.st_ino)
                except OSError:
                    new_key = None
                created = old_key != new_key
                results.append(
                    {
                        "path": target.path,
                        "reopened": True,
                        "new_inode": new_key[1] if new_key else None,
                        "created_new": created,
                    }
                )
        return {"ok": True, "pid": os.getpid(), "results": results}

    # -- writing -----------------------------------------------------------
    def payload(self, target: Target, chunk: int) -> bytes:
        self.seq += 1
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        head = (
            f"{stamp} pid={os.getpid()} name={self.name} seq={self.seq} "
            f"handle={target.path} "
        )
        filler = "x" * max(0, chunk - len(head) - 1)
        return (head + filler + "\n").encode("utf-8")

    def tick(self) -> None:
        cap = self.args.max_mb * 1024 * 1024
        with self.lock:
            for target in self.handles:
                if target.fh is None:
                    continue
                if cap > 0 and target.written >= cap:
                    # Stay quiet once a handle reaches its cap: the demo must
                    # not grow a deleted inode without bound.
                    continue
                chunk = self.payload(target, self.args.chunk)
                target.write(chunk)
                target.written += len(chunk)

    def write_loop(self) -> None:
        while not STOP.is_set():
            self.tick()
            STOP.wait(self.args.interval)

    # -- entry point -------------------------------------------------------
    def run(self) -> int:
        os.makedirs(self.args.log_root, exist_ok=True)
        self.open_all()
        self.register()
        log(
            f"mode={sorted(self.mode)} files={[t.path for t in self.handles]}"
        )
        writer = threading.Thread(target=self.write_loop, name="write", daemon=True)
        writer.start()
        while not STOP.is_set():
            STOP.wait(0.25)

        with self.lock:
            for target in self.handles:
                target.close()
        if self.control is not None:
            try:
                self.control.close()
            except OSError:
                pass
        if self.registered:
            for suffix in (".sock", ".json"):
                path = os.path.join(
                    os.path.dirname(self.registered), self.name + suffix
                )
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        log("exit")
        return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="demo log writer")
    parser.add_argument("--name", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--log-root", default=os.environ.get("LOG_ROOT", "/data/logs"))
    parser.add_argument("--mode", default="single")
    parser.add_argument("--control", default=os.environ.get("STATE_DIR"))
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--max-mb", type=float, default=64.0,
                        help="stop appending to a handle after this many MiB")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="exit after N seconds (0 = forever)")
    args = parser.parse_args(argv)
    if not args.name:
        args.name = f"writer-{os.getpid()}"
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    writer = Writer(args.name, args)
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    if args.duration:
        timer = threading.Timer(args.duration, STOP.set)
        timer.daemon = True
        timer.start()
    return writer.run()


if __name__ == "__main__":
    raise SystemExit(main())
