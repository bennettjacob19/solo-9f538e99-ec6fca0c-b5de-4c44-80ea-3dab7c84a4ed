"""Unit tests for the /proc helpers and configuration."""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

from logreclaim import procfs


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="requires Linux /proc"
)


def test_open_deleted_file_reports_real_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "app.log")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, b"a" * 4096)
            os.fsync(fd)
            before = os.fstat(fd)
            os.unlink(path)
            after = os.fstat(fd)

            files = procfs.open_files(os.getpid())
            mine = [f for f in files if f.fd == fd]
            assert mine, "own descriptor should be visible in /proc"
            item = mine[0]
            assert item.deleted
            assert item.ino == before.st_ino
            assert item.target == path
            assert item.blocks == after.st_blocks
            assert item.allocated == after.st_blocks * 512
        finally:
            os.close(fd)


def test_live_file_is_not_deleted_and_identity_matches_path():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "live.log")
        open(path, "wb").close()
        key = procfs.key_for_path(path)
        assert key is not None
        assert not key.label.startswith("-")
        ok, reason = procfs.path_is_key(path, key)
        assert ok and reason == "ok"
        os.unlink(path)
        ok, reason = procfs.path_is_key(path, key)
        assert not ok and "不存在" in reason


def test_path_is_key_detects_replaced_inode():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rotate.log")
        holder = open(path, "wb")
        try:
            key = procfs.key_for_path(path)
            os.unlink(path)
            # The old inode is still open, so the new file must get a new one.
            open(path, "wb").close()
            assert procfs.key_for_path(path) != key
            ok, reason = procfs.path_is_key(path, key)
            assert not ok and "inode" in reason
        finally:
            holder.close()


def test_proc_identity_distinguishes_reused_pids():
    me = os.getpid()
    ident = procfs.process_identity(me)
    assert ident is not None
    assert ident.pid == me
    assert procfs.process_identity(999_999_999) is None


def test_first_arg_skips_interpreter():
    assert procfs.first_arg(["/usr/bin/python3", "-u", "/app/writer.py", "--x"]) == "/app/writer.py"
    assert procfs.first_arg(["/usr/sbin/nginx", "-g", "daemon off;"]) == "nginx"
    assert procfs.first_arg(["/bin/nginx", "-c", "x.conf"]) == "nginx"


def test_human_bytes_and_mib_round_trip():
    assert procfs.human_bytes(0) == "0 B"
    assert procfs.human_bytes(1536) == "1.5 KiB"
    assert procfs.bytes_to_mib(procfs.mib_to_bytes(3.5)) == 3.5
