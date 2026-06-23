"""File-backed handoff artifacts for sequential optimize_anything compositions."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class HandoffConfig:
    mode: str = "none"
    max_evals: int | None = 100
    max_eval_bytes: int | None = 2_000_000
    include_best_candidate: bool = True

    @classmethod
    def from_value(cls, value: Any) -> "HandoffConfig":
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(mode=value)
        if not isinstance(value, Mapping):
            raise TypeError("adapter.handoff must be a mapping, string, or null")
        mode = str(value.get("mode", "none"))
        max_evals = _optional_int(value.get("max_evals", cls.max_evals))
        max_eval_bytes = _optional_int(value.get("max_eval_bytes", cls.max_eval_bytes))
        include_best_candidate = bool(value.get("include_best_candidate", True))
        return cls(
            mode=mode,
            max_evals=max_evals,
            max_eval_bytes=max_eval_bytes,
            include_best_candidate=include_best_candidate,
        )

    @property
    def enabled(self) -> bool:
        return self.mode not in ("", "none", "false", "False", "0")

    @property
    def rich(self) -> bool:
        return self.mode == "rich"


def collect_stage_handoff(
    *,
    config: HandoffConfig,
    handoff_root: Path,
    evals_dir: Path | None,
    stage_idx: int,
    engine: str,
    eval_start: int,
    eval_end: int,
    best_candidate: str,
    best_score: float,
) -> dict[str, Any]:
    """Create a manifest for one completed sequential stage.

    Metadata carries only paths and small scalars. Full eval records stay on
    disk because ARC traces can be large. We intentionally do not synthesize
    GEPA's ``gepa_state.bin`` here; GEPA should consume prior candidates/traces
    through a future warm-start/import API that preserves GEPAState invariants.

    The ``engine`` key in the returned manifest matches the field
    ``gepa.optimize_anything``'s autoresearch engine reads when it materializes
    prior-stage handoff artifacts.
    """
    stage_dir = handoff_root / f"stage_{stage_idx:02d}_{_safe_name(engine)}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    selected_eval_ids = _bounded_eval_ids(eval_start, eval_end, config.max_evals)
    copied: list[dict[str, Any]] = []
    if config.rich and evals_dir is not None:
        eval_trace_dir = stage_dir / "evals"
        eval_trace_dir.mkdir(exist_ok=True)
        for eval_id in selected_eval_ids:
            source = evals_dir / f"{eval_id}.json"
            if not source.exists():
                continue
            target = eval_trace_dir / source.name
            record = _copy_or_truncate_json(source, target, config.max_eval_bytes)
            copied.append(record)
    else:
        eval_trace_dir = None

    best_candidate_path = None
    if config.include_best_candidate:
        best_candidate_path = stage_dir / "best_candidate.txt"
        best_candidate_path.write_text(best_candidate)

    summary = {
        "stage_idx": stage_idx,
        "engine": engine,
        "best_score": best_score,
        "num_evals": max(0, eval_end - eval_start),
        "eval_start": eval_start,
        "eval_end": eval_end,
        "selected_eval_ids": selected_eval_ids,
        "copied_eval_count": len(copied),
        "copied_evals": copied,
        "best_candidate_path": str(best_candidate_path) if best_candidate_path else None,
        "eval_trace_dir": str(eval_trace_dir) if eval_trace_dir else None,
    }
    summary_path = stage_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    return {
        "schema_version": 1,
        "stage_idx": stage_idx,
        "engine": engine,
        "best_score": best_score,
        "num_evals": summary["num_evals"],
        "summary_path": str(summary_path),
        "best_candidate_path": str(best_candidate_path) if best_candidate_path else None,
        "eval_trace_dir": str(eval_trace_dir) if eval_trace_dir else None,
    }


def _bounded_eval_ids(eval_start: int, eval_end: int, max_evals: int | None) -> list[int]:
    ids = list(range(max(0, eval_start), max(eval_start, eval_end)))
    if max_evals is None or max_evals < 0:
        return ids
    return ids[-max_evals:]


def _copy_or_truncate_json(source: Path, target: Path, max_bytes: int | None) -> dict[str, Any]:
    size = source.stat().st_size
    if max_bytes is None or max_bytes < 0 or size <= max_bytes:
        shutil.copy2(source, target)
        return {"file": target.name, "source": str(source), "bytes": size, "truncated": False}

    try:
        data = json.loads(source.read_text())
    except Exception:
        target.write_text(
            json.dumps(
                {
                    "truncated": True,
                    "source": str(source),
                    "source_bytes": size,
                    "reason": "source exceeded max_eval_bytes and could not be parsed as JSON",
                },
                indent=2,
            )
            + "\n"
        )
        return {"file": target.name, "source": str(source), "bytes": size, "truncated": True}

    for key in ("candidate", "info"):
        if key in data:
            data[key] = _truncate_value(data[key], max_bytes // 2)
    data["_handoff_truncated"] = True
    data["_handoff_source"] = str(source)
    data["_handoff_source_bytes"] = size
    target.write_text(json.dumps(data, indent=2, default=str) + "\n")
    return {"file": target.name, "source": str(source), "bytes": size, "truncated": True}


def _truncate_value(value: Any, max_chars: int) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= max_chars:
        return value
    return {
        "truncated": True,
        "original_chars": len(text),
        "preview": text[:max_chars],
    }


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value) or "backend"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
