"""Read-only inspection of local processes via /proc.

Everything here is deliberately Linux-specific. The core facts we need:

* which regular files a process has open,
* whether those files have been unlinked while still open (``(deleted)``),
* the file identity ``(st_dev, st_ino)`` and its *real* allocated size
  (``st_blocks * 512``, not the apparent length),
* a stable process identity (``pid`` + ``starttime``) so a recycled PID is
  never mistaken for the process a plan was built from.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

SECTOR_BYTES = 512
DELETED_SUFFIX = " (deleted)"


@dataclass(frozen=True, order=True)
class ProcId:
    """A process identity that survives PID reuse checks."""

    pid: int
    starttime: int

    @property
    def label(self) -> str:
        return f"{self.pid}@{self.starttime}"


@dataclass(frozen=True, order=True)
class FileKey:
    """Identity of one file on one device: the dedupe key."""

    dev: int
    ino: int

    @property
    def label(self) -> str:
        return f"{os.major(self.dev)}:{os.minor(self.dev)}/{self.ino}"


@dataclass
class OpenFile:
    """One regular-file descriptor of one process."""

    pid: int
    fd: int
    target: str          # path without the " (deleted)" marker
    deleted: bool
    dev: int
    ino: int
    size: int            # st_size
    blocks: int          # st_blocks, 512-byte sectors (actual allocation)
    nlink: int
    mode: int

    @property
    def key(self) -> FileKey:
        return FileKey(self.dev, self.ino)

    @property
    def allocated(self) -> int:
        return self.blocks * SECTOR_BYTES


@dataclass
class ProcessInfo:
    pid: int
    comm: str
    cmdline: List[str]
    ppid: int
    starttime: int
    uid: int
    state: str
    fds: List[OpenFile] = field(default_factory=list)

    @property
    def ident(self) -> ProcId:
        return ProcId(self.pid, self.starttime)

    @property
    def display(self) -> str:
        name = first_arg(self.cmdline) or self.comm
        return f"{self.pid} · {name}"

    @property
    def alive(self) -> bool:
        return self.state not in ("Z", "X", "x")


def first_arg(cmdline: Sequence[str]) -> str:
    """A concise display name for a command line."""
    parts = [p for p in cmdline if p]
    if not parts:
        return ""
    exe = os.path.basename(parts[0])
    if len(parts) > 1 and re.match(r"^(python[\d.]*|node|perl|ruby|bash|sh)$", exe):
        for part in parts[1:]:
            if not part.startswith("-") and "=" not in part:
                return part
    return exe


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _read_text(path: str) -> str:
    return _read_bytes(path).decode("utf-8", "replace")


def read_stat(pid: int) -> Optional[dict]:
    """Parse /proc/<pid>/stat; comm may contain spaces and parentheses."""
    try:
        raw = _read_text(f"/proc/{pid}/stat")
    except OSError:
        return None
    close = raw.rfind(")")
    open_paren = raw.find("(")
    if close < 0 or open_paren < 0:
        return None
    rest = raw[close + 2 :].split()
    if len(rest) < 20:
        return None
    try:
        return {
            "comm": raw[open_paren + 1 : close],
            "state": rest[0],
            "ppid": int(rest[1]),
            "starttime": int(rest[19]),
        }
    except (ValueError, IndexError):
        return None


def read_cmdline(pid: int) -> List[str]:
    try:
        raw = _read_bytes(f"/proc/{pid}/cmdline")
    except OSError:
        return []
    return [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]


def read_uid(pid: int) -> int:
    try:
        for line in _read_text(f"/proc/{pid}/status").splitlines():
            if line.startswith("Uid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return -1


def list_pids() -> List[int]:
    try:
        names = os.listdir("/proc")
    except OSError:
        return []
    return sorted(int(n) for n in names if n.isdigit())


def process_identity(pid: int) -> Optional[ProcId]:
    st = read_stat(pid)
    if st is None or st["state"] in ("Z", "X", "x"):
        return None
    return ProcId(pid, st["starttime"])


def _describe_fd(pid: int, fd: int) -> Optional[OpenFile]:
    link = f"/proc/{pid}/fd/{fd}"
    try:
        target = os.readlink(link)
        st = os.stat(link)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    deleted = target.endswith(DELETED_SUFFIX)
    clean = target[: -len(DELETED_SUFFIX)] if deleted else target
    return OpenFile(
        pid=pid,
        fd=fd,
        target=clean,
        deleted=deleted,
        dev=st.st_dev,
        ino=st.st_ino,
        size=st.st_size,
        blocks=st.st_blocks,
        nlink=st.st_nlink,
        mode=st.st_mode,
    )


def open_files(pid: int) -> List[OpenFile]:
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return []
    out: List[OpenFile] = []
    for name in fds:
        if name.isdigit():
            item = _describe_fd(pid, int(name))
            if item is not None:
                out.append(item)
    out.sort(key=lambda f: f.fd)
    return out


def list_processes(
    only_pids: Optional[Iterable[int]] = None,
    uid: Optional[int] = None,
    include_self: bool = True,
) -> List[ProcessInfo]:
    """Enumerate live processes with their regular-file descriptors."""
    allowed = set(only_pids) if only_pids is not None else None
    me = os.getpid()
    result: List[ProcessInfo] = []
    for pid in list_pids():
        if allowed is not None and pid not in allowed:
            continue
        if pid == me and not include_self:
            continue
        st = read_stat(pid)
        if st is None or st["state"] in ("Z", "X", "x"):
            continue
        if uid is not None and read_uid(pid) != uid:
            continue
        cmdline = read_cmdline(pid)
        proc = ProcessInfo(
            pid=pid,
            comm=st["comm"],
            cmdline=cmdline,
            ppid=st["ppid"],
            starttime=st["starttime"],
            uid=read_uid(pid),
            state=st["state"],
        )
        proc.fds = open_files(pid)
        result.append(proc)
    return result


def deleted_groups(processes: Sequence[ProcessInfo]) -> Dict[FileKey, List[OpenFile]]:
    """All deleted regular files grouped by (device, inode)."""
    groups: Dict[FileKey, List[OpenFile]] = {}
    for proc in processes:
        for item in proc.fds:
            if item.deleted:
                groups.setdefault(item.key, []).append(item)
    return groups


def deleted_holders(groups: Dict[FileKey, List[OpenFile]]) -> Dict[FileKey, set]:
    return {key: {f.pid for f in fds} for key, fds in groups.items()}


def key_for_path(path: str) -> Optional[FileKey]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return FileKey(st.st_dev, st.st_ino)


def path_is_key(path: str, key: FileKey) -> tuple:
    """Does ``path`` still name the live file ``key``?  Returns (ok, reason)."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return False, "路径已不存在"
    except OSError as exc:
        return False, f"无法 stat 路径：{exc.strerror}"
    if not stat.S_ISREG(st.st_mode):
        return False, "路径不再是普通文件"
    if st.st_dev != key.dev or st.st_ino != key.ino:
        return False, "路径现在指向不同的 inode"
    return True, "ok"


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"


MIB = 1024 * 1024


def bytes_to_mib(n: int) -> float:
    return n / MIB


def mib_to_bytes(mib: float) -> int:
    return int(round(mib * MIB))


def block_report(blocks: int) -> str:
    return f"{human_bytes(blocks * SECTOR_BYTES)} ({blocks} × 512B)"
