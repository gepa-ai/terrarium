"""External bubblewrap jail for ``claude --print`` subprocesses.

Wraps the whole ``claude`` invocation in our own ``bwrap`` namespace instead
of relying on Claude Code's built-in ``sandbox.enabled: true`` settings. The
internal sandbox crashes on Ubuntu 24.04 with
``bwrap: Can't mount tmpfs on /newroot/sbin: No such file or directory``
because it tries to mount tmpfs on top of ``/sbin``, which is a symlink in
the merged-``/usr`` layout. We control the bwrap argv, so we can detect
symlinks and emit ``--symlink`` instead of ``--tmpfs``.

Layout we expose inside the jail:

- ``/usr`` and friends: read-only bind, with symlinks recreated for
  merged-``/usr`` distros (Ubuntu 24.04+, Fedora, Arch). On older Debian /
  RHEL where ``/bin`` is a real directory, those paths get ``--ro-bind``
  instead.
- ``/etc``: only the handful of files needed for DNS, certs, and user
  lookups (``resolv.conf``, ``hosts``, ``passwd``, ``group``, ``ssl``...).
  The surrounding ``/etc`` is bwrap's auto-created tmpfs, so writes to
  ``/etc`` succeed inside the jail but do not leak to the host.
- ``/proc``, ``/dev``, ``/tmp``: standard mounts (``--proc``, ``--dev``,
  fresh ``--tmpfs``).
- ``$HOME/.claude``, ``$HOME/.claude.json``, ``$HOME/.cache``: writable —
  Claude Code stores sessions, config, and caches here.
- ``$HOME/.local``: read-only — ``claude`` itself lives under here.
- ``work_dir``: the only writable path under ``/data``-style trees. Sibling
  run dirs are completely invisible (their parent paths aren't bound, so
  ``ls`` on the parent shows only the path stem leading to ``work_dir``).

Network namespace is shared with the host (no ``--unshare-net``) so the
agent can reach ``localhost:<eval-server-port>`` and ``api.anthropic.com``.
``WebFetch`` / ``WebSearch`` are denied at the tool layer; arbitrary
``curl`` from Bash inside the jail is *not* blocked — acceptable trade for
the GEPA-proposer use case where there's no incentive to phone external
services.

Used by every adapter that spawns ``claude``: ``ClaudeCodeAdapter``,
``MetaHarnessAdapter``, ``ClaudeCodeReflectionProposer``, and
``ClaudeCodeAgentProposer``. Each caller prepends :func:`bwrap_prefix` to
its argv when sandboxed; the rest of the ``claude`` flags
(``--permission-mode bypassPermissions`` and the
:data:`DENY_WEB_TOOLS` ``--disallowedTools=...`` string) are inlined and
unconditional — they don't depend on whether bwrap wraps the call.

TODO(sandbox-leak): binding ``$HOME/.claude`` writable exposes every prior
session transcript under ``~/.claude/projects/<project>/<uuid>.jsonl`` to
the sandboxed agent. Confirmed reproducible: a probe ``grep -r SECRET /``
inside the jail finds markers planted by the parent Claude Code session
because the parent's transcript is in the same projects dir. For GEPA
proposers this is a real cheating channel — sibling proposer transcripts
contain their full code attempts and scores. Fix idea: per-job HOME with a
copy of ``.claude.json`` (auth) and a fresh empty ``.claude/projects/``,
then update ``_copy_session_transcript`` to read from the per-job HOME.
"""

from __future__ import annotations

import os
from pathlib import Path

# System directories we expose read-only. For each, we check at runtime:
# symlink → recreate with ``--symlink``; real dir → ``--ro-bind``; missing
# → skip. This handles both merged-/usr (Ubuntu 24.04+) and split layouts.
_SYSTEM_PATHS: tuple[str, ...] = (
    "/bin", "/sbin", "/lib", "/lib32", "/lib64",
    "/usr/bin", "/usr/sbin", "/usr/lib", "/usr/lib32", "/usr/lib64",
    "/usr/local",
)

# Individual /etc files needed for DNS, certs, and user/group lookups.
# Bound individually so the rest of /etc stays a clean tmpfs in the jail.
_ETC_FILES: tuple[str, ...] = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/passwd",
    "/etc/group",
    "/etc/ld.so.cache",
    "/etc/localtime",
    "/etc/ssl",
    "/etc/ca-certificates",
    "/etc/alternatives",
)

# Always-denied Claude tool flag. WebFetch / WebSearch have no role in GEPA
# and the OS jail can't tell them apart from arbitrary outbound HTTP, so we
# stop them at the Claude layer. Reflection-only callers extend this with
# their own ``--disallowedTools=...`` string instead of layering helpers.
DENY_WEB_TOOLS: str = "--disallowedTools=WebFetch,WebSearch"


def _system_bind_args() -> list[str]:
    """Build ``--ro-bind`` / ``--symlink`` args for standard system dirs."""
    args: list[str] = []
    for path in _SYSTEM_PATHS:
        if os.path.islink(path):
            args.extend(["--symlink", os.readlink(path), path])
        elif os.path.isdir(path):
            args.extend(["--ro-bind", path, path])
    return args


def _etc_bind_args() -> list[str]:
    """Bind the small allow-list of ``/etc`` files read-only."""
    args: list[str] = []
    for path in _ETC_FILES:
        if os.path.exists(path) or os.path.islink(path):
            args.extend(["--ro-bind", path, path])
    return args


def bwrap_prefix(
    work_dir: Path | str,
    *,
    extra_writable: list[Path | str] | None = None,
) -> list[str]:
    """Return the ``bwrap`` argv prefix that jails everything that follows.

    Caller usage::

        cmd = bwrap_prefix(work_dir) + ["claude", "--print", ..., *claude_sandbox_args()]
        subprocess.run(cmd, env=env, capture_output=True, text=True)

    ``--chdir`` is set to ``work_dir`` inside the jail, so ``cwd=`` on the
    subprocess call is unnecessary (but harmless if the caller still sets it
    — it only affects where the bwrap process itself is launched).

    Args:
        work_dir: The single ``/data``-tree path that becomes writable in
            the jail. Its parent directories are auto-created as empty
            stems by bwrap, so siblings stay invisible.
        extra_writable: Additional paths to bind read-write (e.g., a shared
            cache directory).
    """
    home = Path.home()
    work = Path(work_dir).resolve()

    args: list[str] = [
        "bwrap",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        *_system_bind_args(),
        *_etc_bind_args(),
        # Claude config + on-disk session transcripts.
        # See TODO(sandbox-leak) at the top of this file.
        "--bind", str(home / ".claude"), str(home / ".claude"),
        "--bind", str(home / ".claude.json"), str(home / ".claude.json"),
        # Claude binary lives under ~/.local/share/claude; ~/.local/bin/claude
        # is a symlink to it. Read-only is enough.
        "--ro-bind", str(home / ".local"), str(home / ".local"),
        "--bind", str(home / ".cache"), str(home / ".cache"),
        # The sole writable /data path.
        "--bind", str(work), str(work),
        # Scrub hostname so the jail can't be fingerprinted by it.
        "--unshare-uts", "--hostname", "sandbox",
        "--setenv", "HOME", str(home),
        "--chdir", str(work),
    ]
    for p in extra_writable or ():
        resolved = str(Path(p).resolve())
        args.extend(["--bind", resolved, resolved])
    return args


