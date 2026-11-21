"""Registry of demo writer processes and the command channel to them.

Each writer process binds a Unix domain socket and writes a small JSON
registration file into ``<state_dir>/agents``.  The web application only ever
talks to agents it finds through that registry, so the "reopen" action can
never be aimed at an arbitrary process.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .procfs import ProcId, read_stat


class AgentError(RuntimeError):
    pass


@dataclass
class AgentInfo:
    pid: int
    starttime: int
    name: str
    socket: str
    paths: List[str] = field(default_factory=list)
    started_at: float = 0.0

    @property
    def ident(self) -> ProcId:
        return ProcId(self.pid, self.starttime)

    @property
    def display(self) -> str:
        return f"{self.name} · {self.pid}"


@dataclass
class AgentView:
    info: AgentInfo
    alive: bool
    reason: str = ""

    @property
    def ident(self) -> ProcId:
        return self.info.ident


def agents_dir(state_dir: str) -> str:
    return os.path.join(state_dir, "agents")


def read_registry(state_dir: str) -> List[AgentInfo]:
    directory = agents_dir(state_dir)
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    seen: set = set()
    out: List[AgentInfo] = []
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            info = AgentInfo(
                pid=int(data["pid"]),
                starttime=int(data["starttime"]),
                name=str(data.get("name") or f"pid-{data['pid']}"),
                socket=str(data["socket"]),
                paths=[str(p) for p in data.get("paths", [])],
                started_at=float(data.get("started_at", 0.0)),
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if info.ident in seen:
            continue
        seen.add(info.ident)
        out.append(info)
    out.sort(key=lambda a: (a.name, a.pid))
    return out


def verify_registry(state_dir: str) -> List[AgentView]:
    views: List[AgentView] = []
    for info in read_registry(state_dir):
        st = read_stat(info.pid)
        if st is None:
            views.append(AgentView(info, False, "进程已退出"))
            continue
        if st["state"] in ("Z", "X", "x"):
            views.append(AgentView(info, False, "进程已僵尸化"))
            continue
        if st["starttime"] != info.starttime:
            views.append(AgentView(info, False, "PID 已被复用"))
            continue
        if not os.path.exists(info.socket):
            views.append(AgentView(info, False, "控制套接字不存在"))
            continue
        views.append(AgentView(info, True))
    return views


def is_agent_socket(state_dir: str, path: str) -> bool:
    """Defense in depth: only connect to sockets under the agents dir."""
    root = os.path.realpath(agents_dir(state_dir))
    real = os.path.realpath(path)
    return real == root or real.startswith(root + os.sep)


def send_command(
    state_dir: str,
    socket_path: str,
    payload: Dict[str, Any],
    timeout: float = 3.0,
) -> Dict[str, Any]:
    if not is_agent_socket(state_dir, socket_path):
        raise AgentError("拒绝连接：套接字不在 agent 目录内")
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(socket_path)
        sock.sendall(data)
        chunks: List[bytes] = []
        total = 0
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if chunk.endswith(b"\n"):
                break
            if total > 1024 * 1024:
                raise AgentError("响应过长")
    except socket.timeout as exc:
        raise AgentError("等待写入进程响应超时") from exc
    except OSError as exc:
        raise AgentError(f"无法连接写入进程：{exc.strerror or exc}") from exc
    finally:
        sock.close()
    raw = b"".join(chunks).strip()
    if not raw:
        raise AgentError("写入进程未返回内容")
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except ValueError as exc:
        raise AgentError("写入进程返回了无效响应") from exc


def close_agent(info: AgentInfo, state_dir: str) -> None:
    """Best-effort shutdown request when the supervisor is stopping."""
    try:
        send_command(state_dir, info.socket, {"op": "shutdown"}, timeout=1.0)
    except AgentError:
        pass
