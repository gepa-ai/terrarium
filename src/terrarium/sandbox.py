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
- a per-call ``$HOME``: writable, with copied top-level Claude auth/config
  and a fresh empty ``.claude/projects`` directory.
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

Shared ``~/.claude/projects`` and ``~/.cache`` are not bound by default.
Callers must pass explicit ``extra_writable`` paths for any benchmark setting
that allows shared cache access.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Linux uses bwrap; macOS falls back to Claude Code's Seatbelt sandbox
# (see ``claude_settings_args`` below). The bug that motivated the bwrap
# rewrite is Linux-only.
_IS_MACOS = sys.platform == "darwin"

# File tools whitelisted per allowed dir on the Seatbelt path. Includes
# Glob because under ``--permission-mode default`` (which we use to make
# the allowlist enforce — see ``claude_settings_args``) every unlisted
# tool call auto-denies in ``--print`` mode.
_FILE_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob", "Edit", "Write", "NotebookEdit")

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


def prepare_claude_home(base_dir: Path | str) -> Path:
    """Create an isolated HOME for one sandboxed Claude subprocess.

    The home gets only top-level Claude auth/config files plus an empty
    ``.claude/projects`` directory. Project transcripts from the parent home
    are deliberately not copied.
    """
    source_home = Path.home()
    base = Path(base_dir).resolve()
    claude_home = base / ".terrarium_claude_home"
    claude_dir = claude_home / ".claude"
    projects_dir = claude_dir / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)

    source_json = source_home / ".claude.json"
    if source_json.exists():
        shutil.copy2(source_json, claude_home / ".claude.json")

    source_claude = source_home / ".claude"
    if source_claude.is_dir():
        for child in source_claude.iterdir():
            if child.name == "projects" or not child.is_file():
                continue
            try:
                shutil.copy2(child, claude_dir / child.name)
            except OSError:
                pass

    return claude_home


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
    claude_home: Path | str | None = None,
    extra_writable: list[Path | str] | None = None,
) -> list[str]:
    """Return the ``bwrap`` argv prefix that jails everything that follows.
    Returns ``[]`` on macOS — that platform uses :func:`claude_settings_args`
    as a fallback because ``bwrap`` is Linux-only.

    Caller usage (works on both platforms)::

        cmd = bwrap_prefix(work_dir)               # Linux: bwrap argv. macOS: [].
        cmd += ["claude", "--print", ...]
        cmd += claude_settings_args(work_dir)      # macOS: --settings JSON. Linux: [].
        cmd += ["--permission-mode", ..., DENY_WEB_TOOLS]
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
    if _IS_MACOS:
        # No bwrap on macOS — caller should use claude_settings_args() instead.
        return []

    real_home = Path.home()
    home = Path(claude_home).resolve() if claude_home is not None else prepare_claude_home(work_dir)
    work = Path(work_dir).resolve()

    args: list[str] = [
        "bwrap",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        *_system_bind_args(),
        *_etc_bind_args(),
        # Per-call Claude config + empty projects dir. Do not bind the parent
        # ~/.claude/projects transcript history into the sandbox.
        "--bind", str(home), str(home),
        # Claude binary lives under ~/.local/share/claude; ~/.local/bin/claude
        # is a symlink to it. Read-only is enough.
        "--ro-bind", str(real_home / ".local"), str(real_home / ".local"),
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


# macOS fallback: Claude Code's built-in Seatbelt sandbox via --settings JSON.
# bwrap is Linux-only; the bug that motivated upstream's bwrap rewrite (tmpfs
# on /sbin under merged-/usr) is Linux-only, so Seatbelt is fine here.


def _abs_glob(path: str) -> str:
    """Format an absolute path as Claude's ``//<path>/**`` rule pattern."""
    return f"/{path}/**"


def _build_macos_sandbox_settings(
    work_dir: Path | str,
    *,
    extra_writable: list[Path | str] | None = None,
) -> dict[str, Any]:
    """Settings JSON for Claude Code's Seatbelt sandbox.

    Two layers: ``sandbox.filesystem.*`` confines Bash subprocesses;
    ``permissions.allow`` whitelists file tools (only enforces under
    ``--permission-mode default``, see :func:`claude_settings_args`).

    The two layers cover different attack surfaces. The tool layer
    (``permissions.allow``) is the primary read barrier — agent Read /
    Glob / Edit calls are whitelisted to work_dir only. Seatbelt is
    secondary, for Bash subprocesses where the tool layer doesn't
    apply: it denies ``~/`` (project state, auto-memory, sibling
    proposer outputs) but leaves system paths readable so claude itself
    + standard utilities (curl, jq, openssl, python) work.
    """
    work_paths = [str(Path(work_dir).resolve())]
    work_paths.extend(str(Path(p).resolve()) for p in extra_writable or ())

    # Bash subprocesses need /tmp + /private/tmp writable for claude's
    # per-call script staging dir (/tmp/claude-<uid>/cwd-XXXX). Both
    # forms because Seatbelt path-matches literally and /tmp is a
    # symlink to /private/tmp.
    write_paths = work_paths + ["/tmp", "/private/tmp"]

    allow_rules: list[str] = [
        f"{tool}({_abs_glob(p)})" for p in work_paths for tool in _FILE_TOOLS
    ]
    allow_rules.append("Bash(*)")
    return {
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": False,
            # Disable the documented dangerouslyDisableSandbox escape
            # hatch (https://code.claude.com/docs/en/sandboxing). Without
            # this, Bash failures inside the sandbox get retried outside
            # the sandbox after the agent invokes that flag — a real
            # bypass channel for proposer-context isolation.
            "allowUnsandboxedCommands": False,
            "network": {
                # eval.sh hits the terrarium EvalServer on localhost.
                # macOS Seatbelt blocks localhost binding by default
                # (curl exit 7 — failed to connect); allowLocalBinding
                # re-enables it. allowedDomains stays empty so the
                # agent's bash can't phone external services (no
                # cheating channel via api.anthropic.com etc; claude
                # itself bypasses the bash sandbox for its own calls).
                # macOS-only flag; Linux bwrap shares the host network
                # namespace and doesn't need this.
                "allowLocalBinding": True,
            },
            "filesystem": {
                # The documented canonical pattern (per the Sandboxing
                # docs): block reads under home, leave system paths
                # readable. Default sandbox read behavior is "entire
                # computer except denied dirs" — denying ~/ covers every
                # real leak channel (auto-memory at ~/.claude/, sibling
                # proposer outputs at ~/Downloads/<repo>/outputs/, repo
                # source at ~/Downloads/<repo>/src/) without blocking
                # /usr/bin, /private/etc/ssl, /System/Library that
                # claude itself + standard utilities need.
                "denyRead":  ["~/"],
                "allowRead":  work_paths,
                # No explicit denyWrite — the documented default is
                # "writes only to cwd and its subdirs" (claude is
                # launched with cwd=work_dir). allowWrite extends that
                # default to /tmp for claude's session staging.
                "allowWrite": write_paths,
            },
        },
        "permissions": {"allow": allow_rules},
    }


def claude_settings_args(
    work_dir: Path | str,
    *,
    extra_writable: list[Path | str] | None = None,
) -> list[str]:
    """Settings + permission flags for the macOS Seatbelt path. Empty on
    Linux (bwrap_prefix already confines).

    Includes ``--permission-mode default`` so the ``permissions.allow``
    whitelist in the settings JSON actually enforces (under
    bypassPermissions the allowlist is advisory and unrelated tool calls
    succeed). In ``--print`` mode any unlisted tool call auto-denies
    because there's no human to approve the prompt — so the allowlist
    becomes a strict tool-layer whitelist that complements the
    OS-level Seatbelt confinement.
    """
    if not _IS_MACOS:
        return []
    settings = _build_macos_sandbox_settings(work_dir, extra_writable=extra_writable)
    return [
        "--settings", json.dumps(settings),
        "--permission-mode", "default",
    ]

