"""Lightweight sandboxing for ``claude --print`` subprocesses.

Builds the settings-JSON payload and CLI flags that confine a spawned
``claude`` session to a specific working directory. Two layers:

- **OS sandbox** (``sandbox.filesystem.*``) — bubblewrap on Linux, Seatbelt
  on macOS. Wraps **Bash tool calls only**: shell commands spawned by the
  session can read/write only inside ``work_dir`` (+ ``extra_dirs``).
  See: https://code.claude.com/docs/en/sandboxing

- **Tool permission allow-list** (``permissions.allow``) — whitelists the
  content-bearing file tools (Read/Grep/Edit/Write/NotebookEdit) to
  ``work_dir``. Because we **do not** pass ``--permission-mode
  bypassPermissions``, Claude falls into the default mode, which prompts
  for any unlisted tool call — and those prompts auto-deny in ``--print``
  (no human to approve). So allow-only acts as a strict whitelist; no
  deny rules are necessary.
  Glob is left unrestricted: empirically it ignores the permission rules
  and the OS sandbox, but it only returns file *names* (no content), so
  leaking directory listings outside ``work_dir`` is an acceptable loss.
  See: https://code.claude.com/docs/en/permissions#read-and-edit

Used by every adapter that spawns ``claude``: ``ClaudeCodeAdapter``,
``MetaHarnessAdapter``, and ``GEPAAdapter``'s ``ClaudeCodeReflectionProposer``.
Each caller just appends the output of :func:`sandbox_args` to its own
``claude --print`` argv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Tools we always deny: evolution is local, no reason to browse.
_ALWAYS_DENIED_TOOLS: tuple[str, ...] = ("WebFetch", "WebSearch")

# Content-bearing file tools we whitelist per allowed directory. Glob is
# deliberately excluded — it auto-allows regardless of rules, but only
# returns filenames (no content), so listing escape is accepted.
_FILE_TOOLS: tuple[str, ...] = ("Read", "Grep", "Edit", "Write", "NotebookEdit")


def _abs_glob(path: str) -> str:
    """Format an absolute path as Claude's ``//<path>/**`` rule pattern."""
    return f"/{path}/**"


def build_sandbox_settings(
    work_dir: Path | str,
    *,
    extra_dirs: list[Path | str] | None = None,
    allow_network: bool = True,
    allow_bash: bool = True,
) -> dict[str, Any]:
    """Return a settings.json dict that confines ``claude`` to ``work_dir``.

    Args:
        work_dir: Sole directory the agent can read/write by default.
        extra_dirs: Additional directories to allow (e.g., ``/tmp``).
        allow_network: When False, disables outbound network at the sandbox
            layer. Leave True for adapters that POST to the local EvalServer.
        allow_bash: When False, omits the ``Bash(*)`` allow rule so every
            shell invocation is auto-denied. Use for LM-only flows that
            have no reason to run commands.
    """
    paths = [str(Path(work_dir).resolve())]
    paths.extend(str(Path(p).resolve()) for p in extra_dirs or ())

    allow_rules: list[str] = []
    for p in paths:
        for tool in _FILE_TOOLS:
            allow_rules.append(f"{tool}({_abs_glob(p)})")
    if allow_bash:
        allow_rules.append("Bash(*)")

    settings: dict[str, Any] = {
        "sandbox": {
            "enabled": True,
            # Degrade gracefully when bubblewrap/seatbelt isn't available
            # instead of hard-failing the whole run.
            "failIfUnavailable": False,
            "filesystem": {
                "denyRead": ["//", "~/"],
                "allowRead": paths,
                "denyWrite": ["//", "~/"],
                "allowWrite": paths,
            },
        },
        "permissions": {"allow": allow_rules},
    }
    if not allow_network:
        settings["sandbox"]["network"] = {"allowedDomains": []}
    return settings


def sandbox_args(
    work_dir: Path | str,
    *,
    extra_dirs: list[Path | str] | None = None,
    allow_network: bool = True,
    allow_bash: bool = True,
    deny_tools: list[str] | None = None,
) -> list[str]:
    """CLI args to append to a ``claude --print`` invocation.

    The caller must **not** also pass ``--permission-mode bypassPermissions``
    — doing so disables the prompt-driven default-deny that makes the
    allow-list act as a whitelist.

    Three mechanisms stacked:

    - ``--settings <json>`` installs the per-session sandbox + file-tool
      allow list from :func:`build_sandbox_settings`.
    - ``--disallowedTools`` always denies ``WebFetch``/``WebSearch`` plus
      any ``deny_tools`` supplied by the caller (de-duped, comma-joined).
    """
    settings = build_sandbox_settings(
        work_dir,
        extra_dirs=extra_dirs,
        allow_network=allow_network,
        allow_bash=allow_bash,
    )
    disallow: list[str] = list(_ALWAYS_DENIED_TOOLS)
    for t in deny_tools or ():
        if t not in disallow:
            disallow.append(t)
    return [
        "--settings", json.dumps(settings),
        f"--disallowedTools={','.join(disallow)}",
    ]
