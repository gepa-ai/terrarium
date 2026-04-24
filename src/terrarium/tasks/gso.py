"""GSO performance optimization benchmark: one terrarium task per instance.

Each instance in the GSO benchmark (118 instances) is registered as an
independent single-task — because each instance targets a specific
open-source repo and the candidate is a unified diff that must be applied
to that repo's specific source tree.

Task naming:
    gso_<instance_id>   e.g. gso_numpy__numpy-ee75c87
    gso_smoke           curated 3-instance subset for pipeline tests
    gso_subset          curated ~30-instance subset of known-good instances

Data source:
    Instance metadata is pulled from the HuggingFace dataset
    ``gso-bench/gso`` (cached by the ``datasets`` library).

Evaluation:
    Each eval spins up a Docker container from the pre-built GSO image
    (``slimshetty/gso:gso.eval.<arch>.<instance_id>``), applies the
    candidate patch via ``git apply``, runs install commands, executes
    test scripts, and computes speedup metrics vs baseline and expert.

    Prerequisites::

        pip install 'terrarium[gso]'
        # Docker must be installed and running.

    Override the GSO metrics repo location::

        export GSO_DIR=/path/to/existing/clone

Usage::

    python -m terrarium task.name=gso_numpy__numpy-ee75c87 adapter=gepa
    python -m terrarium task=gso_smoke adapter=claude_code
    python -m terrarium task=gso_subset adapter=gepa budget.max_evals=30
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GSO_DATASET = "gso-bench/gso"
_GSO_REPO_URL = "https://github.com/gso-bench/gso.git"

_INITIAL_CANDIDATE = ""  # empty diff = baseline (no changes)

_OBJECTIVE = (
    "Optimize the performance of code in an open-source repository by "
    "producing a unified diff (patch). The goal is to match or exceed the "
    "expert developer's optimization (opt_commit = true)."
)

_SOURCE_CONTEXT_LIMIT = 100_000  # max bytes of source files to include in background

# Curated subset: ~31 instances from repos that run well in standard settings.
# Excludes: llama-cpp-python (C++ compilation), transformers (very large images),
# pillow-simd (redundant with Pillow).
_SUBSET_IDS: tuple[str, ...] = (
    # numpy (8 of 36)
    "numpy__numpy-09db9c7",
    "numpy__numpy-22ab9aa",
    "numpy__numpy-330057f",
    "numpy__numpy-567b57d",
    "numpy__numpy-728fedc",
    "numpy__numpy-83c780d",
    "numpy__numpy-ba89ef9",
    "numpy__numpy-ee75c87",
    # pandas (8 of 34)
    "pandas-dev__pandas-061c2e9",
    "pandas-dev__pandas-191557d",
    "pandas-dev__pandas-233bd83",
    "pandas-dev__pandas-2cdca01",
    "pandas-dev__pandas-438b957",
    "pandas-dev__pandas-609c3b7",
    "pandas-dev__pandas-84aca21",
    "pandas-dev__pandas-e7e3676",
    # Pillow (all 4)
    "python-pillow__Pillow-63f398b",
    "python-pillow__Pillow-d8af3fc",
    "python-pillow__Pillow-f854676",
    "python-pillow__Pillow-fd8ee84",
    # pydantic (all 4)
    "pydantic__pydantic-4a09447",
    "pydantic__pydantic-ac9e6ee",
    "pydantic__pydantic-addf1f9",
    "pydantic__pydantic-c2647ab",
    # tornado (all 4)
    "tornadoweb__tornado-1b464c4",
    "tornadoweb__tornado-4d4c1e0",
    "tornadoweb__tornado-9a18f6c",
    "tornadoweb__tornado-ac13ee5",
    # tokenizers (2 of 4)
    "huggingface__tokenizers-076319d",
    "huggingface__tokenizers-bfd9cde",
    # datasets (1 of 3)
    "huggingface__datasets-5994036",
)

# Smoke test: 3 pydantic instances (pure Python, small images, fast tests).
_SMOKE_IDS: tuple[str, ...] = (
    "pydantic__pydantic-4a09447",
    "pydantic__pydantic-ac9e6ee",
    "pydantic__pydantic-addf1f9",
)

# ---------------------------------------------------------------------------
# GSO repo management — clone once for metrics imports
# ---------------------------------------------------------------------------


def _gso_cache_dir() -> Path:
    """Return the GSO repo path (persistent cache location by default)."""
    env = os.environ.get("GSO_DIR")
    if env:
        return Path(env).expanduser()
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "terrarium" / "gso-bench"


def _ensure_gso_repo() -> Path:
    """Clone the GSO repo if needed and make it importable.

    Uses a file lock to prevent races when multiple terrarium processes
    start concurrently.
    """
    gso_dir = _gso_cache_dir()
    src_dir = gso_dir / "src" / "gso"

    if src_dir.exists():
        _add_gso_to_path(gso_dir)
        return gso_dir

    if shutil.which("git") is None:
        raise RuntimeError(
            "git is required to clone the GSO metrics repo but was not found on PATH. "
            f"Install git, or manually clone {_GSO_REPO_URL} to {gso_dir} "
            "and set GSO_DIR."
        )

    gso_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = gso_dir.parent / ".terrarium_gso_clone.lock"

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            if src_dir.exists():
                _add_gso_to_path(gso_dir)
                return gso_dir

            print(f"[terrarium] Cloning GSO repo into {gso_dir} (one-time)...", flush=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", _GSO_REPO_URL, str(gso_dir)],
                check=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"git clone of {_GSO_REPO_URL} failed ({e}). "
                f"Clone manually to {gso_dir} and/or set GSO_DIR."
            ) from e
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    _add_gso_to_path(gso_dir)
    return gso_dir


def _add_gso_to_path(gso_dir: Path) -> None:
    src = str(gso_dir / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


_gso_modules: dict | None = None


def _get_gso_imports() -> dict:
    """Lazy import GSO modules after repo is available."""
    global _gso_modules
    if _gso_modules is not None:
        return _gso_modules

    _ensure_gso_repo()

    from gso.constants import MIN_PROB_SPEEDUP, OPT_THRESH
    from gso.harness.grading.evalscript import MAX_ITERS, MAX_TIME
    from gso.harness.grading.metrics import get_opt_status
    from gso.harness.environment.patches import apply_patches

    _gso_modules = {
        "MIN_PROB_SPEEDUP": MIN_PROB_SPEEDUP,
        "OPT_THRESH": OPT_THRESH,
        "MAX_ITERS": MAX_ITERS,
        "MAX_TIME": MAX_TIME,
        "get_opt_status": get_opt_status,
        "apply_patches": apply_patches,
    }
    return _gso_modules


# ---------------------------------------------------------------------------
# Docker helpers — simple subprocess wrappers
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    """Check if Docker CLI is available and the daemon is running."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _docker_run(image: str, *, name: str | None = None) -> str:
    """Start a detached Docker container. Returns the container ID.

    Pulls the image if not present locally (may be slow on first use).
    """
    cmd = ["docker", "run", "-d", "-w", "/testbed"]
    if name:
        cmd.extend(["--name", name])
    cmd.extend([image, "sleep", "infinity"])

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1200,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker run failed for {image}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _docker_exec(
    container_id: str,
    command: str,
    *,
    cwd: str | None = None,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run a command inside a container. Returns {output, returncode}."""
    cmd = ["docker", "exec"]
    if cwd:
        cmd.extend(["-w", cwd])
    cmd.extend([container_id, "bash", "-lc", command])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return {
            "output": result.stdout + result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"output": f"Command timed out after {timeout}s", "returncode": -1}


def _docker_cp(src: str, dst: str) -> None:
    """Copy files between host and container."""
    subprocess.run(
        ["docker", "cp", src, dst],
        capture_output=True, timeout=120, check=True,
    )


def _docker_rm(container_id: str) -> None:
    """Stop and remove a container."""
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass  # best-effort cleanup


# ---------------------------------------------------------------------------
# HuggingFace dataset loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _gso_instances() -> dict[str, dict[str, Any]]:
    """Load the GSO HF dataset; return instances keyed by instance_id.

    Cached — the HF ``datasets`` library also caches the parquet on disk.
    """
    from datasets import load_dataset

    ds = load_dataset(_GSO_DATASET, split="test")
    rows: dict[str, dict[str, Any]] = {}
    for row in ds:
        iid = row["instance_id"]
        rows[iid] = dict(row)
    return rows


# ---------------------------------------------------------------------------
# Docker image helper
# ---------------------------------------------------------------------------


def _get_docker_image(instance: dict) -> str:
    """Return the pre-built GSO Docker image name for an instance."""
    arch = instance.get("arch", "x86_64")
    iid = instance["instance_id"].lower()
    return f"slimshetty/gso:gso.eval.{arch}.{iid}"


# ---------------------------------------------------------------------------
# Test script generation and injection
# ---------------------------------------------------------------------------


def _get_test_scripts(instance: dict) -> list[str]:
    """Return patched test scripts from the instance (using GSO's patches)."""
    gso = _get_gso_imports()
    tests = instance.get("tests", [])
    if isinstance(tests, str):
        tests = [tests]
    return gso["apply_patches"](instance["instance_id"], list(tests))


def _get_test_count(instance: dict) -> int:
    tests = instance.get("tests", [])
    return len(tests) if isinstance(tests, list) else 1


def _get_install_commands(instance: dict) -> list[str]:
    """Get install commands, filtering out ``git clean -xfd``."""
    cmds = instance.get("install_commands", [])
    if isinstance(cmds, str):
        cmds = [cmds]
    return [c for c in cmds if c != "git clean -xfd"]


def _script_preamble() -> list[str]:
    return [
        "#!/bin/bash",
        "set -euo pipefail",
        "cd /testbed",
        "source .venv/bin/activate",
        "echo 'setuptools<82' > /tmp/uv_build_constraints.txt",
        "export UV_BUILD_CONSTRAINT=/tmp/uv_build_constraints.txt",
        "export HF_HUB_DISABLE_XET=1",
        'if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; fi',
    ]


def _test_run_lines(instance: dict, prefix_tag: str, flag: str) -> list[str]:
    gso = _get_gso_imports()
    repo = instance.get("repo", "")
    iters = gso["MAX_ITERS"](repo)
    timeout = gso["MAX_TIME"](repo)
    test_count = _get_test_count(instance)
    lines: list[str] = []
    for i in range(test_count):
        for _ in range(iters):
            tp = f"timeout {timeout}s " if timeout > 0 else ""
            lines.append(
                f'{tp}python /gso_test_{i}.py '
                f'{prefix_tag}_{i}.txt {flag} --file_prefix gso_{i}'
            )
    lines.append('echo ">>>>> RESULTS"')
    for i in range(test_count):
        lines.append(f'echo ">>>>> Test {i}"')
        lines.append(f"cat {prefix_tag}_{i}.txt")
    return lines


def _make_run_tests_script(instance: dict, mode: str = "eqcheck") -> str:
    """Generate bash to run tests inside the container.

    mode: "reference" for base timing, "eqcheck" for patch timing.
    """
    flag = "--reference" if mode == "reference" else "--eqcheck"
    prefix_tag = "base" if mode == "reference" else "result"

    lines = _script_preamble()

    if mode == "eqcheck":
        install_cmds = _get_install_commands(instance)
        if install_cmds:
            lines.extend(install_cmds)

    lines.extend(_test_run_lines(instance, prefix_tag, flag))
    return "\n".join(lines) + "\n"


def _make_commit_test_script(instance: dict) -> str:
    """Generate bash to run tests on the expert commit."""
    opt_commit = instance.get("opt_commit", "")
    repo = instance.get("repo", "")

    lines = _script_preamble()

    lines.extend([
        f'git remote add origin https://github.com/{repo}.git 2>/dev/null || true',
        f'git fetch origin {opt_commit}',
        'git clean -xfd || true',
        f'git checkout {opt_commit}',
    ])

    install_cmds = _get_install_commands(instance)
    if install_cmds:
        lines.extend(install_cmds)

    lines.extend(_test_run_lines(instance, "commit", "--reference"))
    return "\n".join(lines) + "\n"


def _inject_test_scripts(container_id: str, instance: dict) -> None:
    """Copy patched test scripts into the container."""
    scripts = _get_test_scripts(instance)
    with tempfile.TemporaryDirectory() as td:
        for i, code in enumerate(scripts):
            path = Path(td) / f"gso_test_{i}.py"
            path.write_text(code, encoding="utf-8")
            _docker_cp(str(path), f"{container_id}:/gso_test_{i}.py")


def _cleanup_test_scripts(container_id: str, instance: dict) -> None:
    """Remove test scripts from the container."""
    n = _get_test_count(instance)
    files = " ".join(f"/gso_test_{i}.py" for i in range(n))
    _docker_exec(container_id, f"rm -f {files}", timeout=10)


# ---------------------------------------------------------------------------
# Timing parsing
# ---------------------------------------------------------------------------

_EXEC_TIME_RE = re.compile(r"Execution time:\s+([\d.]+)s")


def _parse_timing(raw_output: str, test_count: int) -> list[list[float]]:
    """Parse execution times from test output.

    Looks for ``>>>>> Test N`` markers followed by
    ``Execution time: X.XXXs`` lines.
    """
    result: list[list[float]] = [[] for _ in range(test_count)]
    current_test = -1
    for line in raw_output.splitlines():
        line = line.strip()
        if line.startswith(">>>>> Test "):
            try:
                current_test = int(line.split()[-1])
            except ValueError:
                pass
            continue
        m = _EXEC_TIME_RE.match(line)
        if m and 0 <= current_test < test_count:
            result[current_test].append(float(m.group(1)))
    return result


# ---------------------------------------------------------------------------
# Speedup computation — delegates to GSO's get_opt_status
# ---------------------------------------------------------------------------


def _compute_speedup(
    base_times: list[list[float]],
    patch_times: list[list[float]],
    commit_times: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Compute speedup metrics using GSO's official evaluation logic."""
    gso = _get_gso_imports()

    if not any(t for t in base_times) or not any(t for t in patch_times):
        return {
            "error": "no valid timing data",
            "gm_speedup": 0,
            "opt_base": False,
            "opt_commit": False,
        }

    time_map = {
        "base_times": base_times,
        "patch_times": patch_times,
        "commit_times": (
            commit_times
            if commit_times and any(t for t in commit_times)
            else []
        ),
        "main_times": None,
    }

    try:
        result = gso["get_opt_status"](time_map)
    except Exception as e:
        logger.error("get_opt_status failed: %s", e)
        return {
            "error": str(e),
            "gm_speedup": 0,
            "opt_base": False,
            "opt_commit": False,
        }

    opt_stats = result["opt_stats"]
    return {
        "gm_speedup": opt_stats.get("gm_speedup_patch_base") or 0,
        "hm_speedup": opt_stats.get("hm_speedup_patch_base") or 0,
        "hm_speedup_vs_commit": (
            opt_stats.get("hm_speedup_patch_commit")
            or opt_stats.get("hm_speedup_vs_commit")
            or 0
        ),
        "opt_base": result["opt_base"],
        "opt_commit": result["opt_commit"],
        "base_mean": result["time_stats"].get("base_mean") or 0,
        "patch_mean": result["time_stats"].get("patch_mean") or 0,
        "time_stats": result["time_stats"],
        "opt_stats": opt_stats,
    }


# ---------------------------------------------------------------------------
# Base / commit timing cache
# ---------------------------------------------------------------------------

_base_timing_cache: dict[str, dict[str, Any]] = {}
_base_timing_locks: dict[str, threading.Lock] = {}
_base_timing_global_lock = threading.Lock()


def _get_base_lock(instance_id: str) -> threading.Lock:
    with _base_timing_global_lock:
        if instance_id not in _base_timing_locks:
            _base_timing_locks[instance_id] = threading.Lock()
        return _base_timing_locks[instance_id]


def _get_base_timing(instance: dict) -> dict[str, Any]:
    """Compute and cache base + commit timing for an instance.

    Spins up a fresh container on first call, computes baseline and expert
    timing, caches the result, then destroys the container. Thread-safe:
    concurrent callers for the same instance block until the first finishes.
    """
    instance_id = instance["instance_id"]

    lock = _get_base_lock(instance_id)
    with lock:
        if instance_id in _base_timing_cache:
            return _base_timing_cache[instance_id]

        image = _get_docker_image(instance)
        container_name = f"terrarium-gso-base-{instance_id[:30]}-{uuid.uuid4().hex[:8]}"
        container_id = None

        try:
            logger.info("Computing base timing for %s ...", instance_id)
            container_id = _docker_run(image, name=container_name)
            test_count = _get_test_count(instance)
            _inject_test_scripts(container_id, instance)

            # Base timing
            base_script = _make_run_tests_script(instance, mode="reference")
            base_result = _run_script_in_container(container_id, base_script, "gso_base_eval.sh")
            base_times = _parse_timing(base_result, test_count)
            logger.info(
                "Base timing for %s: %s",
                instance_id,
                [round(sum(t) / len(t), 4) for t in base_times if t],
            )

            # Commit (expert) timing
            base_commit = instance.get("base_commit", "")
            opt_commit = instance.get("opt_commit", "")
            if opt_commit and opt_commit != base_commit:
                logger.info("Computing commit timing for %s ...", instance_id)
                commit_script = _make_commit_test_script(instance)
                commit_result = _run_script_in_container(
                    container_id, commit_script, "gso_commit_eval.sh",
                )
                commit_times = _parse_timing(commit_result, test_count)
                logger.info(
                    "Commit timing for %s: %s",
                    instance_id,
                    [round(sum(t) / len(t), 4) for t in commit_times if t],
                )
            else:
                commit_times = base_times

            _cleanup_test_scripts(container_id, instance)

            cached = {"base_times": base_times, "commit_times": commit_times}
            _base_timing_cache[instance_id] = cached
            return cached

        finally:
            if container_id:
                _docker_rm(container_id)


def _run_script_in_container(
    container_id: str, script: str, script_name: str,
) -> str:
    """Write a script into the container, run it, clean up, return output."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        f.flush()
        _docker_cp(f.name, f"{container_id}:/tmp/{script_name}")
    os.unlink(f.name)

    result = _docker_exec(container_id, f"bash /tmp/{script_name}", cwd="/testbed", timeout=3600)
    _docker_exec(container_id, f"rm -f /tmp/{script_name}", timeout=10)
    return result["output"]


# ---------------------------------------------------------------------------
# Source context extraction
# ---------------------------------------------------------------------------

_source_context_cache_dir: Path | None = None


def _get_source_cache_dir() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = cache_root / "terrarium" / "gso-source-context"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_source_context(instance: dict) -> str:
    """Extract key source files from the Docker image for the task background.

    Starts a temporary container, finds relevant source files in /testbed,
    and returns a concatenated string of their contents (limited to ~100KB).
    Caches results on disk.

    Returns an empty string (with a warning) if Docker is unavailable.
    """
    instance_id = instance["instance_id"]

    # Check disk cache
    cache_dir = _get_source_cache_dir()
    cache_file = cache_dir / f"{instance_id}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    if not _docker_available():
        warnings.warn(
            f"Docker not available — cannot extract source context for {instance_id}. "
            "The task background will not include source code.",
            stacklevel=2,
        )
        return ""

    image = _get_docker_image(instance)
    container_name = f"terrarium-gso-src-{instance_id[:30]}-{uuid.uuid4().hex[:8]}"
    container_id = None

    try:
        container_id = _docker_run(image, name=container_name)

        # Find relevant source files — exclude tests, venv, git, pycache
        find_result = _docker_exec(
            container_id,
            (
                "find /testbed -maxdepth 4 "
                "\\( -name '*.py' -o -name '*.c' -o -name '*.cpp' "
                "-o -name '*.pyx' -o -name '*.pxd' \\) "
                "-not -path '*/__pycache__/*' "
                "-not -path '*/.venv/*' "
                "-not -path '*/.git/*' "
                "-not -path '*/test*' "
                "-not -path '*/doc*' "
                "-not -path '*/example*' "
                "-not -path '*/benchmark*' "
                "2>/dev/null | head -100"
            ),
            timeout=30,
        )

        files = [
            f.strip() for f in find_result["output"].splitlines()
            if f.strip() and f.strip().startswith("/testbed")
        ]

        if not files:
            cache_file.write_text("", encoding="utf-8")
            return ""

        # Prioritize files mentioned in prob_script or hints_text
        prob_script = instance.get("prob_script", "") or ""
        hints_text = instance.get("hints_text", "") or ""
        context_text = prob_script + " " + hints_text

        def priority(fpath: str) -> int:
            basename = os.path.basename(fpath)
            name_no_ext = os.path.splitext(basename)[0]
            # Higher priority (lower number) for files mentioned in context
            if basename in context_text or name_no_ext in context_text:
                return 0
            # Medium priority for __init__.py and core modules
            if basename == "__init__.py":
                return 2
            # Lower priority for everything else
            return 1

        files.sort(key=priority)

        # Read files, respecting the size limit
        parts: list[str] = []
        total_bytes = 0
        for fpath in files:
            if total_bytes >= _SOURCE_CONTEXT_LIMIT:
                break
            cat_result = _docker_exec(container_id, f"cat '{fpath}'", timeout=10)
            content = cat_result["output"]
            if not content or cat_result["returncode"] != 0:
                continue
            # Skip very large files (>10KB each)
            if len(content) > 10_000:
                continue
            parts.append(f"### {fpath}\n```\n{content}\n```")
            total_bytes += len(content)

        source_context = "\n\n".join(parts)
        cache_file.write_text(source_context, encoding="utf-8")
        return source_context

    except Exception as e:
        logger.warning("Source extraction failed for %s: %s", instance_id, e)
        return ""

    finally:
        if container_id:
            _docker_rm(container_id)


# ---------------------------------------------------------------------------
# Background builder
# ---------------------------------------------------------------------------


def _build_task_text(instance: dict) -> str:
    """Build the problem description from prob_script + hints."""
    parts: list[str] = []
    prob = (instance.get("prob_script") or "").strip()
    if prob:
        parts.append(prob)
    hints = (instance.get("hints_text") or "").strip()
    if hints:
        parts.append("## Hints\n\n" + hints)
    return "\n\n".join(parts)


def _build_background(instance: dict, source_context: str) -> str:
    """Build the full background string for the task."""
    repo = instance.get("repo", "unknown")
    task_text = _build_task_text(instance)

    sections = [
        f"## Performance Optimization Task\n\n"
        f"You are optimizing code in the repository `{repo}` to improve performance.",
        f"## Performance Benchmark\n\n{task_text}" if task_text else "",
    ]

    if source_context:
        sections.append(
            "## Relevant Source Code\n\n"
            "The following source files are from the repository at /testbed in the "
            "Docker container. Your candidate must be a unified diff (patch) that "
            "can be applied with `git apply` to this codebase.\n\n"
            + source_context
        )

    sections.append(
        "## Candidate Format\n\n"
        "Your candidate MUST be a unified diff (the output of `git diff`). Example:\n\n"
        "```diff\n"
        "--- a/path/to/file.py\n"
        "+++ b/path/to/file.py\n"
        "@@ -10,7 +10,7 @@\n"
        " existing context line\n"
        "-old line to replace\n"
        "+new optimized line\n"
        " existing context line\n"
        "```\n\n"
        "The diff will be applied to the repository at /testbed using `git apply`.\n"
        "After applying, install commands will run, then performance tests will execute.\n"
        "An empty candidate (no changes) scores ~1.0 (baseline performance)."
    )

    sections.append(
        "## Scoring\n\n"
        "- **Score** = geometric mean speedup over baseline (higher is better)\n"
        "- `opt_base` = true if speedup >= 1.2x over baseline (necessary but not sufficient)\n"
        "- `opt_commit` = true when you match the expert developer's optimization "
        "(>= 0.95x of their harmonic-mean speedup) — **this is the goal**\n"
        "- If the patch fails to apply or tests error, score is 0.0\n"
        "- `hm_speedup_vs_commit` shows your speedup relative to the expert"
    )

    return "\n\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------


def _apply_patch(container_id: str, candidate: str) -> tuple[bool, str]:
    """Apply a unified diff candidate in the container.

    Tries multiple strategies (strict, then lenient). Returns (success, message).
    """
    if not candidate.strip():
        return True, "empty patch (baseline)"

    # Write diff to temp file and copy into container
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f:
        f.write(candidate)
        f.flush()
        _docker_cp(f.name, f"{container_id}:/tmp/patch.diff")
    os.unlink(f.name)

    # Try applying with progressively more lenient options
    strategies = [
        "git apply /tmp/patch.diff",
        "git apply --ignore-space-change /tmp/patch.diff",
        "git apply --ignore-space-change --reject /tmp/patch.diff",
        "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
    ]
    for strategy in strategies:
        result = _docker_exec(container_id, strategy, cwd="/testbed", timeout=60)
        if result["returncode"] == 0:
            return True, f"applied via: {strategy}"

    return False, result["output"][:2000]


def _evaluate(candidate: str, *, instance: dict) -> tuple[float, dict[str, Any]]:
    """Evaluate a unified diff candidate on one GSO instance.

    Never raises — apply failures, test errors, and Docker issues all come
    back as ``(0.0, info)`` with diagnostic fields so reflection-based
    adapters can read the error and adjust.
    """
    instance_id = instance["instance_id"]
    image = _get_docker_image(instance)

    # Check Docker availability
    if not _docker_available():
        return 0.0, {
            "score": 0.0,
            "instance_id": instance_id,
            "status": "docker_not_available",
            "message": (
                "Docker is not installed or not running. "
                "GSO evaluation requires Docker."
            ),
        }

    # Get cached base timing (may create a temporary container)
    try:
        base_data = _get_base_timing(instance)
        if not base_data.get("base_times") or not any(
            t for t in base_data["base_times"]
        ):
            return 0.0, {
                "score": 0.0,
                "instance_id": instance_id,
                "status": "base_timing_failed",
                "message": "Base timing computation returned no valid timing data.",
            }
    except Exception as e:
        return 0.0, {
            "score": 0.0,
            "instance_id": instance_id,
            "status": "base_timing_failed",
            "message": f"Base timing failed: {e}",
        }

    # Create eval container
    container_name = (
        f"terrarium-gso-eval-{instance_id[:20]}-{uuid.uuid4().hex[:8]}"
    )
    container_id = None

    try:
        container_id = _docker_run(image, name=container_name)
    except Exception as e:
        return 0.0, {
            "score": 0.0,
            "instance_id": instance_id,
            "status": "image_pull_failed",
            "message": f"Failed to start container from {image}: {e}",
        }

    try:
        # Inject test scripts
        try:
            _inject_test_scripts(container_id, instance)
        except Exception as e:
            return 0.0, {
                "score": 0.0,
                "instance_id": instance_id,
                "status": "error",
                "message": f"Failed to inject test scripts: {e}",
            }

        # Apply patch
        applied, apply_msg = _apply_patch(container_id, candidate)
        if not applied:
            return 0.0, {
                "score": 0.0,
                "instance_id": instance_id,
                "status": "patch_apply_failed",
                "message": f"Patch could not be applied: {apply_msg}",
            }

        # Run test scripts (eqcheck mode includes install commands)
        test_script = _make_run_tests_script(instance, mode="eqcheck")
        try:
            test_output = _run_script_in_container(
                container_id, test_script, "gso_patch_eval.sh",
            )
        except Exception as e:
            return 0.0, {
                "score": 0.0,
                "instance_id": instance_id,
                "status": "tests_failed",
                "message": f"Test execution failed: {e}",
            }

        # Parse timing
        test_count = _get_test_count(instance)
        patch_times = _parse_timing(test_output, test_count)

        if not any(t for t in patch_times):
            return 0.0, {
                "score": 0.0,
                "instance_id": instance_id,
                "status": "tests_failed",
                "message": "No timing data parsed from test output.",
                "raw_output": test_output[:3000],
            }

        # Compute speedup
        try:
            metrics = _compute_speedup(
                base_data["base_times"],
                patch_times,
                commit_times=base_data.get("commit_times"),
            )
        except Exception as e:
            return 0.0, {
                "score": 0.0,
                "instance_id": instance_id,
                "status": "scoring_error",
                "message": f"Speedup computation failed: {e}",
            }

        if "error" in metrics:
            return 0.0, {
                "score": 0.0,
                "instance_id": instance_id,
                "status": "scoring_error",
                "message": f"Scoring error: {metrics['error']}",
            }

        score = float(metrics.get("gm_speedup", 0))
        return score, {
            "score": score,
            "instance_id": instance_id,
            "status": "success",
            "gm_speedup": metrics.get("gm_speedup", 0),
            "hm_speedup": metrics.get("hm_speedup", 0),
            "opt_base": metrics.get("opt_base", False),
            "opt_commit": metrics.get("opt_commit", False),
            "hm_speedup_vs_commit": metrics.get("hm_speedup_vs_commit", 0),
            "base_mean": metrics.get("base_mean", 0),
            "patch_mean": metrics.get("patch_mean", 0),
            "apply_message": apply_msg,
        }

    finally:
        if container_id:
            _docker_rm(container_id)


# ---------------------------------------------------------------------------
# Task factories
# ---------------------------------------------------------------------------


def _make_instance_task(instance_id: str) -> Task:
    """Factory for a single-instance task. Runs on first get_task() access."""
    rows = _gso_instances()
    if instance_id not in rows:
        raise KeyError(
            f"GSO instance {instance_id!r} not found in HF dataset."
        )
    instance = rows[instance_id]

    source_context = _extract_source_context(instance)
    background = _build_background(instance, source_context)

    def eval_fn(candidate: str) -> tuple[float, dict[str, Any]]:
        return _evaluate(candidate, instance=instance)

    return Task(
        name=f"gso_{instance_id}",
        objective=_OBJECTIVE,
        background=background,
        initial_candidate=_INITIAL_CANDIDATE,
        eval_fn=eval_fn,
        metadata={
            "type": "single_task",
            "candidate_type": "diff",
            "benchmark": "gso",
            "instance_id": instance_id,
            "repo": instance.get("repo", ""),
            "arch": instance.get("arch", "x86_64"),
            "docker_image": _get_docker_image(instance),
        },
    )


def _make_smoke_task() -> Task:
    """Curated 3-instance subset for pipeline validation.

    Uses pydantic instances (pure Python, small images, fast tests).
    Note: a single candidate (diff) can't generalize across instances
    (they target different repos). This task is for plumbing validation only.
    """
    rows = _gso_instances()
    available = [iid for iid in _SMOKE_IDS if iid in rows]
    if not available:
        raise RuntimeError(
            f"None of the smoke IDs {_SMOKE_IDS} are in the GSO HF dataset. "
            f"Available: {sorted(rows)[:10]}..."
        )

    examples = [
        Example(
            id=iid,
            inputs={
                "instance_id": iid,
                "repo": rows[iid].get("repo", ""),
                "prob_script": (rows[iid].get("prob_script") or "")[:500],
            },
            expected=None,
        )
        for iid in available
    ]

    def eval_fn(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
        instance = rows[example.inputs["instance_id"]]
        return _evaluate(candidate, instance=instance)

    descriptions = "\n\n---\n\n".join(
        f"### {iid}\n\n{_build_task_text(rows[iid])[:500]}"
        for iid in available
    )
    return Task(
        name="gso_smoke",
        objective=_OBJECTIVE,
        background=descriptions,
        initial_candidate=_INITIAL_CANDIDATE,
        eval_fn=eval_fn,
        train_set=examples,
        metadata={
            "type": "generalization",
            "candidate_type": "diff",
            "benchmark": "gso",
        },
    )


def _make_subset_task() -> Task:
    """Curated ~33-instance subset of known-good GSO instances.

    Covers numpy, pandas, pillow, pydantic, tornado, tokenizers, and datasets.
    Excludes repos with complex builds or very large images.
    """
    rows = _gso_instances()
    available = [iid for iid in _SUBSET_IDS if iid in rows]
    if not available:
        raise RuntimeError(
            f"None of the subset IDs are in the GSO HF dataset. "
            f"Available: {sorted(rows)[:10]}..."
        )

    examples = [
        Example(
            id=iid,
            inputs={
                "instance_id": iid,
                "repo": rows[iid].get("repo", ""),
                "prob_script": (rows[iid].get("prob_script") or "")[:500],
            },
            expected=None,
        )
        for iid in available
    ]

    def eval_fn(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
        instance = rows[example.inputs["instance_id"]]
        return _evaluate(candidate, instance=instance)

    return Task(
        name="gso_subset",
        objective=_OBJECTIVE,
        background=(
            "Curated subset of GSO performance optimization tasks. "
            f"Covers {len(available)} instances across "
            "numpy, pandas, pillow, pydantic, tornado, tokenizers, and datasets."
        ),
        initial_candidate=_INITIAL_CANDIDATE,
        eval_fn=eval_fn,
        train_set=examples,
        metadata={
            "type": "generalization",
            "candidate_type": "diff",
            "benchmark": "gso",
            "subset_size": len(available),
        },
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register_all() -> None:
    """Register the smoke task, subset task, and one factory per GSO instance.

    Registration is cheap: each factory captures just the instance_id. The
    HF dataset isn't loaded until a factory is actually invoked (via
    get_task()), and then it's loaded exactly once (cached).

    If the ``datasets`` package isn't installed, we skip silently — the
    GSO extra wasn't installed, and other terrarium tasks should still work.
    """
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        return

    try:
        rows = _gso_instances()
    except Exception as e:
        warnings.warn(
            f"Could not load GSO instance list from HuggingFace: {e}. "
            "GSO tasks will not be registered. "
            "Other terrarium tasks are unaffected.",
            stacklevel=2,
        )
        return

    register_task_factory("gso_smoke", _make_smoke_task)
    register_task_factory("gso_subset", _make_subset_task)
    for iid in rows:
        register_task_factory(
            f"gso_{iid}",
            lambda i=iid: _make_instance_task(i),
        )


_register_all()
