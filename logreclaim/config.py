"""Runtime configuration, driven by environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_path(name: str, default: str) -> str:
    return os.path.abspath(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    log_root: str
    state_dir: str
    secret_key: str
    host: str
    port: int
    demo_enabled: bool
    max_writers: int
    plan_ttl_seconds: int
    write_chunk: int
    write_interval: float
    scan_scope: str = "agents"

    @property
    def log_roots(self):
        roots = [self.log_root]
        extra = os.environ.get("LOG_EXTRA_ROOTS", "")
        for item in extra.split(","):
            item = item.strip()
            if item:
                roots.append(os.path.abspath(item))
        return roots

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            log_root=_env_path("LOG_ROOT", "/data/logs"),
            state_dir=_env_path("STATE_DIR", "/data/state"),
            secret_key=os.environ.get("SECRET_KEY", "logreclaim-dev-secret"),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8080")),
            demo_enabled=os.environ.get("DEMO_ENABLED", "1") not in ("0", "false", "no"),
            max_writers=min(8, int(os.environ.get("MAX_WRITERS", "8"))),
            plan_ttl_seconds=int(os.environ.get("PLAN_TTL_SECONDS", "900")),
            write_chunk=int(os.environ.get("WRITE_CHUNK", "16384")),
            write_interval=float(os.environ.get("WRITE_INTERVAL", "0.4")),
            scan_scope=os.environ.get("SCAN_SCOPE", "agents"),
        )
