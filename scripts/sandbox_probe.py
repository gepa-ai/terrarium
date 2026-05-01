"""Probe Terrarium's Claude subprocess sandbox.

This script does not launch Claude. It verifies the local Linux bwrap setup
that wraps Claude subprocesses:

- per-call HOME has an empty .claude/projects directory
- parent ~/.cache is not bound into the jail by default
- repo-root files outside the adapter work dir are not readable
- sibling run output directories are not readable
- prior Claude transcripts are not readable
- hidden test files or serialized task data outside work_dir are not readable
- work_dir remains read/write
- localhost eval-server traffic still works

Run from the repo root:

    .venv/bin/python scripts/sandbox_probe.py
"""

from __future__ import annotations

import http.server
import shutil
import socketserver
import subprocess
import tempfile
import threading
from pathlib import Path

from terrarium.sandbox import bwrap_prefix, prepare_claude_home


class _ProbeHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        if self.path == "/health":
            body = b"terrarium-probe-ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_eval_probe_server() -> tuple[socketserver.TCPServer, str]:
    server = socketserver.TCPServer(("127.0.0.1", 0), _ProbeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/health"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="terrarium_sandbox_probe_") as tmp:
        root = Path(tmp)
        repo_root = root / "repo"
        output_root = root / "outputs"
        work_dir = output_root / "run_current" / "agent_work"
        sibling_dir = output_root / "run_previous"
        hidden_dir = output_root / "hidden_task_data"
        cache_dir = root / "home_cache" / ".cache" / "huggingface"
        transcript_dir = root / "home_claude" / ".claude" / "projects" / "old-run"

        work_dir.mkdir(parents=True)
        repo_root.mkdir()
        sibling_dir.mkdir(parents=True)
        hidden_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        transcript_dir.mkdir(parents=True)

        (repo_root / "repo_secret.txt").write_text("REPO_SECRET\n")
        (sibling_dir / "secret.txt").write_text("SECRET\n")
        (hidden_dir / "test_set.json").write_text('{"hidden": true}\n')
        (cache_dir / "token").write_text("CACHE_SECRET\n")
        (transcript_dir / "session.jsonl").write_text('{"secret": true}\n')
        (work_dir / "visible.txt").write_text("VISIBLE\n")

        claude_home = prepare_claude_home(work_dir)
        projects = claude_home / ".claude" / "projects"
        assert projects.is_dir(), "isolated Claude home must contain .claude/projects"
        assert not list(projects.iterdir()), "isolated .claude/projects must start empty"

        prefix = bwrap_prefix(work_dir, claude_home=claude_home)
        if not prefix or shutil.which("bwrap") is None:
            print("sandbox home ok; bwrap probe skipped")
            return

        server, eval_url = _start_eval_probe_server()
        script = (
            "set -euo pipefail\n"
            f"test -f {work_dir / 'visible.txt'}\n"
            "printf sandbox-write > sandbox_write.txt\n"
            "test -f sandbox_write.txt\n"
            f"test ! -e {repo_root / 'repo_secret.txt'}\n"
            f"test ! -e {sibling_dir / 'secret.txt'}\n"
            f"test ! -e {hidden_dir / 'test_set.json'}\n"
            f"test ! -e {cache_dir / 'token'}\n"
            f"test ! -e {transcript_dir / 'session.jsonl'}\n"
            f"test ! -e {Path.home() / '.cache'}\n"
            "test -d \"$HOME/.claude/projects\"\n"
            "test -z \"$(find \"$HOME/.claude/projects\" -mindepth 1 -print -quit)\"\n"
            "/usr/bin/python3 - <<'PY'\n"
            "import urllib.request\n"
            f"body = urllib.request.urlopen({eval_url!r}, timeout=5).read()\n"
            "assert body == b'terrarium-probe-ok', body\n"
            "PY\n"
        )
        try:
            subprocess.run([*prefix, "bash", "-lc", script], check=True)
        finally:
            server.shutdown()
            server.server_close()

    print("sandbox probe ok")


if __name__ == "__main__":
    main()
