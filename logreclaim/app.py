"""Flask UI for Linux deleted-log space reclamation."""

from __future__ import annotations

import math
import os
import time
from typing import Optional

from flask import Flask, current_app, flash, jsonify, redirect, render_template, request, url_for

from .config import Config
from .controller import Controller
from .procfs import block_report, human_bytes, mib_to_bytes


def _controller() -> Controller:
    return current_app.extensions["logreclaim"]


def create_app(config: Optional[Config] = None) -> Flask:
    cfg = config or Config.from_env()
    os.makedirs(cfg.state_dir, exist_ok=True)
    os.makedirs(cfg.log_root, exist_ok=True)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = cfg.secret_key
    app.config["LOGRECLAIM"] = cfg
    controller = Controller(cfg)
    app.extensions["logreclaim"] = controller

    def _fmt_bytes(value: int) -> str:
        return f"{human_bytes(value)} · {value:,} B"

    app.jinja_env.filters["bytes"] = _fmt_bytes
    app.jinja_env.filters["blocks"] = block_report
    app.jinja_env.filters["mib"] = lambda b: f"{b / (1024 * 1024):.2f}"
    app.jinja_env.filters["ts"] = lambda value: time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(value)
    )

    @app.context_processor
    def inject_globals():
        return {"config": cfg}

    @app.get("/")
    def index():
        controller = _controller()
        # In host scope this returns the controlled-writer first screen
        # immediately and kicks off a background full-host scan.
        view = controller.index_view()
        return render_template(
            "index.html",
            view=view,
            host_status=view.host_status,
            plan=None,
            snapshot=None,
            target_mib="",
            error=None,
        )

    @app.get("/scan/status")
    def scan_status():
        controller = _controller()
        status = controller.host_scan_status()
        return jsonify(
            {
                "state": status.state,
                "table_epoch": status.table_epoch,
                "attempt": status.attempt,
                "running": status.is_running,
                "failed": status.is_failed,
                "host_wide": status.table_is_host_wide,
                "error": status.error,
                "started_at": status.started_at,
                "finished_at": status.finished_at,
            }
        )

    @app.post("/scan/retry")
    def scan_retry():
        controller = _controller()
        if controller.config.scan_scope != "host":
            flash("当前不是全机扫描模式，无需重试。", "warning")
            return redirect(url_for("index"))
        controller.retry_host_scan()
        flash("已重新开始全机核对。", "ok")
        return redirect(url_for("index"))

    @app.post("/plan")
    def plan_view():
        controller = _controller()
        # Planning always reads the machine in real time: never reuse a
        # background table that could predate a rotation or reclamation.
        view = controller.scan()
        host_status = view.host_status
        raw = (request.form.get("target_mib") or "").strip()
        error = None
        plan = None
        target_mib = raw
        try:
            target_mib_value = float(raw)
            if not math.isfinite(target_mib_value) or target_mib_value <= 0:
                raise ValueError
        except ValueError:
            error = "请输入大于 0 的 MiB 数值。"
            return render_template(
                "index.html",
                view=view,
                host_status=host_status,
                plan=None,
                snapshot=None,
                target_mib=raw,
                error=error,
            ), 400

        target_bytes = mib_to_bytes(target_mib_value)
        plan = controller.make_plan(view, target_bytes)
        snapshot = None
        if plan.feasible:
            snapshot = controller.snapshot(view, plan, target_mib_value)
        return render_template(
            "index.html",
            view=view,
            host_status=host_status,
            plan=plan,
            snapshot=snapshot,
            target_mib=raw,
            error=error,
        )

    @app.post("/confirm/<token>")
    def confirm(token: str):
        controller = _controller()
        snap = controller.take_snapshot(token)
        if snap is None:
            flash("计划已过期或不存在，请重新计算。", "warning")
            return redirect(url_for("index"))
        # execute() performs its own real-time full scan before touching
        # anything, and bumps the table epoch afterwards.
        report = controller.execute(snap)
        return render_template("report.html", report=report, snap=snap), (
            409 if not report.executed else 200
        )

    @app.post("/demo/rotate")
    def demo_rotate():
        if not cfg.demo_enabled:
            flash("演示操作已禁用。", "warning")
            return redirect(url_for("index"))
        moved = _controller().demo_rotate()
        if moved:
            flash(
                f"已轮转 {len(moved)} 个日志文件（写入进程仍持有原 inode）。",
                "ok",
            )
        else:
            flash("没有可轮转的日志文件。", "warning")
        return redirect(url_for("index"))

    @app.post("/demo/delete")
    def demo_delete():
        if not cfg.demo_enabled:
            flash("演示操作已禁用。", "warning")
            return redirect(url_for("index"))
        deleted = _controller().demo_delete_old()
        if deleted:
            total = sum(size for _, size in deleted)
            flash(
                f"已删除 {len(deleted)} 个旧日志，释放目录项；"
                f"被进程持有的 {human_bytes(total)} 仍占磁盘。",
                "ok",
            )
        else:
            flash("没有可删除的旧日志。", "warning")
        return redirect(url_for("index"))

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.errorhandler(404)
    def not_found(_exc):
        return render_template("error.html", code=404, message="页面不存在"), 404

    @app.errorhandler(500)
    def server_error(_exc):
        return render_template("error.html", code=500, message="服务器内部错误"), 500

    return app


def main() -> None:
    cfg = Config.from_env()
    application = create_app(cfg)
    application.run(host=cfg.host, port=cfg.port, threaded=True)


if __name__ == "__main__":
    main()
