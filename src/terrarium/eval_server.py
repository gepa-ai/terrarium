"""Eval server: the single choke point for all evaluation and budget enforcement.

Every adapter — in-process (GEPA) or external (Claude Code) — goes through
the EvalServer. It provides two interfaces to the same budget-controlled eval:

1. ``server.evaluate(candidate)`` — direct Python call (no HTTP overhead).
2. ``POST server.url/evaluate`` — HTTP endpoint for external/black-box systems.

The runner always creates the EvalServer. Adapters cannot construct their own.

HTTP Protocol
-------------
POST /evaluate
    Body: {"candidate": "<text>", "example_id": "<optional>"}
    Response: {"score": 1.23, "info": {...}, "budget": {"used": 5, "remaining": 95, ...}}
    Status 429 when budget exhausted.

POST /evaluate_examples
    Body: {"candidate": "<text>", "example_ids": ["a","b","c"]}
    Or:   {"candidate": "<text>", "split": "train"}    # only "train" allowed via HTTP
    Evaluates the candidate on the specified examples (or all in train) in
    parallel, respecting the concurrency limit. Each example = 1 budget tick.
    Response: {"average_score": 1.23, "scores": {"a": 0.9, ...}, "budget": {...}}
    For single-task problems (no dataset), behaves like /evaluate.
    Status 400 when split is not "train" or example_ids include val/test ids.
    Status 429 when budget is insufficient to evaluate all requested examples.

    Val is reachable only via /validate (aggregate-only). Test is not
    reachable from HTTP — only the in-process Python API can score on it,
    which the runner uses for held-out reporting.

GET /status
    Response: {"budget": {...}, "task": "<name>", "best_score": 1.23, "total_evals": 5, "total_cost": 0.42}

GET /task
    Response: {"name": "...", "objective": "...", "background": "...", "initial_candidate": "...", ...}
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from terrarium.budget import BudgetExhausted, BudgetTracker
from terrarium.task import Example, Task
from terrarium.tracking import TerrariumTracker

DEFAULT_MAX_CONCURRENCY = 8


class EvalServer:
    """Eval server with budget enforcement — the single source of truth.

    In-process adapters call :meth:`evaluate` directly.
    External adapters POST to the HTTP endpoint at :attr:`url`.
    Both go through the same budget counter.

    Args:
        task: The task to evaluate.
        budget: Budget tracker for limiting evaluations.
        tracker: Optional experiment tracker for logging.
        max_concurrency: Maximum number of parallel evaluations.
            Controls the thread pool size for batch/parallel eval methods.
            Defaults to 8.
        output_dir: If set, each eval is persisted as ``<dir>/evals/<i>.json``
            (0-indexed) with full score/info/candidate for later analysis,
            alongside a progress ``summary.json``.
    """

    def __init__(
        self,
        task: Task,
        budget: BudgetTracker,
        tracker: TerrariumTracker | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        output_dir: str | Path | None = None,
    ) -> None:
        self.task = task
        self.budget = budget
        self.tracker = tracker
        self.max_concurrency = max_concurrency
        self.best_score: float = float("-inf")
        self.best_candidate: str = task.initial_candidate
        self.best_visible_score: float = float("-inf")
        self.best_visible_candidate: str = task.initial_candidate
        self.best_visible_source: str | None = None
        self.best_validated_score: float = float("-inf")
        self.best_validated_candidate: str | None = None
        self.total_cost: float = 0.0
        self.eval_log: list[dict[str, Any]] = []
        self._start_time: float = time.time()
        self._best_val_score: float = float("-inf")
        self._progress_log: list[dict[str, Any]] = []
        self._candidate_registry: dict[str, int] = {}
        self._next_candidate_id: int = 0
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self.output_dir: Path | None = None
        self._evals_dir: Path | None = None
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._evals_dir = self.output_dir / "evals"
            self._evals_dir.mkdir(exist_ok=True)
        self._eval_semaphore = threading.Semaphore(max_concurrency)

        # Build public example lookup for dataset tasks. Hidden test examples
        # are evaluated only by the runner after search.
        self._examples: dict[str, Example] = {}
        for dataset in [task.train_set, task.val_set]:
            if dataset:
                for ex in dataset:
                    self._examples[ex.id] = ex

        self._pool = ThreadPoolExecutor(max_workers=max_concurrency)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── Direct Python API (used by in-process adapters like GEPA) ───────

    def evaluate(self, candidate: str, example: Example | None = None) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate with budget enforcement.

        This is the primary API for in-process adapters. Same budget counter
        as the HTTP endpoint — there is only one counter.

        Raises:
            BudgetExhausted: When the budget has been used up.
        """
        self._eval_semaphore.acquire()
        try:
            self.budget.check()

            if example is not None:
                score, info = self.task.eval_fn(candidate, example)
            else:
                score, info = self.task.eval_fn(candidate)

            self.budget.record(score)
            self._track(candidate, score, info)

            info["_budget"] = self.budget.status()
            return score, info
        finally:
            self._eval_semaphore.release()

    def evaluate_examples(
        self,
        candidate: str,
        example_ids: list[str] | None = None,
        split: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate on specific examples or an entire split, in parallel.

        For single-task problems (no dataset), delegates to evaluate().

        Provide either ``example_ids`` (explicit list) or ``split`` ("train"/"val"/"all").
        If both are given, ``example_ids`` takes precedence.

        Returns:
            (average_score, info_dict) with per-example scores and partial-eval metadata.
        """
        if not self.task.has_dataset:
            score, info = self.evaluate(candidate)
            return score, {
                "scores": {"_single": score},
                "infos": {"_single": info},
                "num_evaluated": 1,
                "num_total": 1,
                "partial": False,
                "_budget": self.budget.status(),
            }

        # Resolve examples
        examples: list[Example] = []
        if example_ids is not None:
            if not example_ids:
                raise ValueError("example_ids must not be empty")
            unknown_ids: list[str] = []
            for eid in example_ids:
                if eid in self._examples:
                    examples.append(self._examples[eid])
                else:
                    unknown_ids.append(eid)
            if unknown_ids:
                preview = ", ".join(unknown_ids[:5])
                raise ValueError(f"Unknown or hidden example IDs: {preview}")
        elif split is not None:
            if split == "test":
                raise ValueError("test split is hidden during search")
            if split not in ("train", "val", "all"):
                raise ValueError("split must be one of: train, val, all")
            if split in ("train", "all") and self.task.train_set:
                examples.extend(self.task.train_set)
            if split in ("val", "all") and self.task.val_set:
                examples.extend(self.task.val_set)
        else:
            if self.task.train_set:
                examples.extend(self.task.train_set)

        if not examples:
            if split in ("train", "val"):
                raise ValueError(f"{split} split is not available during search")
            raise ValueError("No visible examples are available during search")

        if self.budget.remaining is not None and self.budget.remaining < len(examples):
            raise BudgetExhausted(
                f"Not enough budget to evaluate all examples: {self.budget.remaining} remaining, {len(examples)} needed"
            )

        scores: dict[str, float] = {}
        infos: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}

        def _eval_one(ex: Example) -> tuple[str, float, dict[str, Any] | None, str | None]:
            try:
                score, info = self.evaluate(candidate, ex)
                return (ex.id, score, info, None)
            except Exception as e:
                return (ex.id, 0.0, None, str(e))

        futures = {self._pool.submit(_eval_one, ex): ex for ex in examples}

        for future in as_completed(futures):
            eid, score, ex_info, err = future.result()
            if err is not None:
                errors[eid] = err
            else:
                scores[eid] = score
                if ex_info is not None:
                    infos[eid] = ex_info

        all_scores = {**scores, **{eid: 0.0 for eid in errors}}
        avg = sum(all_scores.values()) / len(all_scores)
        info: dict[str, Any] = {
            "scores": all_scores,
            "infos": infos,
            "num_evaluated": len(examples),
            "_budget": self.budget.status(),
        }
        if errors:
            info["errors"] = errors

        if example_ids is None and split in ("train", "all"):
            self.record_visible_candidate(avg, candidate, source=f"{split}_aggregate")

        return avg, info


    def validate(self, candidate: str) -> dict[str, Any]:
        """Evaluate candidate on the full visible val_set, return aggregate score only."""
        if not self.task.val_set:
            raise ValueError("validate() requires a task with val_set")

        val_ids = [ex.id for ex in self.task.val_set]
        avg_score, _ = self.evaluate_examples(candidate, example_ids=val_ids)
        self.record_visible_candidate(avg_score, candidate, source="validation")

        return self.log_progress(avg_score, candidate=candidate)

    def record_visible_candidate(self, score: float, candidate: str, *, source: str) -> None:
        """Track the best aggregate visible-data candidate.

        Per-example evals update ``best_candidate`` for debugging, but benchmark
        submission for dataset tasks should use aggregate train/validation
        selection. This method is the shared bookkeeping path for full train
        aggregate and validation aggregate scores.
        """
        with self._lock:
            if score > self.best_visible_score:
                self.best_visible_score = float(score)
                self.best_visible_candidate = candidate
                self.best_visible_source = source
            if source == "validation" and score > self.best_validated_score:
                self.best_validated_score = float(score)
                self.best_validated_candidate = candidate

    def log_progress(self, val_score: float, candidate: str | None = None, reflection_cost: float = 0.0) -> dict[str, Any]:
        """Record a progress checkpoint."""
        # Register candidate outside _lock to avoid _lock → _io_lock nesting.
        candidate_id: int | None = None
        if candidate is not None:
            candidate_id = self._register_candidate(candidate)

        with self._lock:
            if val_score > self._best_val_score:
                self._best_val_score = val_score
            entry: dict[str, Any] = {
                "val_score": val_score,
                "best_val_score": self._best_val_score,
                "total_evals": self.budget.used,
                "wall_time": time.time() - self._start_time,
                "total_cost": self.total_cost,
                "reflection_cost": reflection_cost,
            }
            if candidate_id is not None:
                entry["candidate_id"] = candidate_id
            self._progress_log.append(entry)

        if self.output_dir is not None:
            with self._io_lock:
                self._append_progress_log(entry)

        return {"val_score": val_score, "best_val_score": self._best_val_score}

    def _register_candidate(self, candidate: str) -> int:
        """Return a stable integer ID for a candidate, registering it if new."""
        if candidate in self._candidate_registry:
            return self._candidate_registry[candidate]
        cid = self._next_candidate_id
        self._next_candidate_id += 1
        self._candidate_registry[candidate] = cid
        if self.output_dir is not None:
            with self._io_lock:
                with open(self.output_dir / "candidates.jsonl", "a") as f:
                    f.write(json.dumps({"candidate_id": cid, "candidate": candidate}) + "\n")
        return cid

    @property
    def progress_log(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._progress_log)

    # ── HTTP server (used by external/black-box adapters) ───────────────

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("Server not started")
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def start(self, port: int = 0) -> int:
        """Start the HTTP server. Returns the port number."""
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/evaluate":
                    server_ref._handle_evaluate(self)
                elif self.path == "/evaluate_examples":
                    server_ref._handle_evaluate_examples(self)
                elif self.path == "/validate":
                    server_ref._handle_validate(self)
                else:
                    self.send_error(404)

            def do_GET(self):
                if self.path == "/status":
                    server_ref._handle_status(self)
                elif self.path == "/task":
                    server_ref._handle_task_info(self)
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass  # suppress request logging

        self._server = HTTPServer(("localhost", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            if self._thread:
                self._thread.join(timeout=1)
            self._server = None
            self._thread = None
        self._pool.shutdown(wait=False)

    # ── Internal ────────────────────────────────────────────────────────

    def _track(self, candidate: str, score: float, info: dict[str, Any] | None = None) -> None:
        cost = float(info.get("cost", 0.0)) if info else 0.0
        with self._lock:
            self.total_cost += cost
            idx = len(self.eval_log)  # 0-indexed file name for per-eval JSON
            entry: dict[str, Any] = {
                "eval": self.budget.used,
                "score": score,
                "candidate_len": len(candidate),
                "wall_time": time.time() - self._start_time,
                "cumulative_cost": self.total_cost,
            }
            if cost:
                entry["cost"] = cost
            self.eval_log.append(entry)
            if score > self.best_score:
                self.best_score = score
                self.best_candidate = candidate
            if self.tracker:
                self.tracker.log_eval(self.budget.used, score, self.best_score, cost)
            snapshot = self._snapshot()

        if self.output_dir is not None:
            with self._io_lock:
                self._write_summary(snapshot)
                self._write_eval_record(idx, entry, candidate, info)

    def _snapshot(self) -> dict[str, Any]:
        """Capture current run state. Must be called inside ``_lock``."""
        return {
            "best_score": self.best_score,
            "best_visible_score": self.best_visible_score,
            "best_visible_source": self.best_visible_source,
            "best_validated_score": self.best_validated_score,
            "total_evals": self.budget.used,
            "total_cost": self.total_cost,
            "budget": self.budget.status(),
        }

    def _append_progress_log(self, entry: dict[str, Any]) -> None:
        assert self.output_dir is not None
        with open(self.output_dir / "progress_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _write_summary(self, snapshot: dict[str, Any]) -> None:
        """Atomically write a lightweight progress snapshot to summary.json.

        Does NOT include eval_log (already in ``evals/<i>.json``) or
        best_candidate (can be large).  The runner writes the complete
        summary at the end.
        """
        assert self.output_dir is not None
        summary_path = self.output_dir / "summary.json"
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, indent=2, default=str))
        tmp.replace(summary_path)

    def _write_eval_record(
        self,
        idx: int,
        entry: dict[str, Any],
        candidate: str,
        info: dict[str, Any] | None,
    ) -> None:
        """Write the full per-eval record (candidate + info) to ``evals/{idx}.json``."""
        assert self._evals_dir is not None
        record = {
            **entry,
            "timestamp": time.time(),
            "candidate": candidate,
            "info": info,
        }
        try:
            (self._evals_dir / f"{idx}.json").write_text(json.dumps(record, indent=2, default=str))
        except Exception:
            # Never let logging failure break an eval.
            pass

    def _train_ids(self) -> set[str]:
        """Ids the HTTP layer is allowed to score. val is reachable only via
        /validate (aggregate); test is unreachable from HTTP entirely."""
        if not self.task.train_set:
            return set()
        return {ex.id for ex in self.task.train_set}

    def _handle_evaluate(self, handler: BaseHTTPRequestHandler) -> None:
        content_length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(content_length)) if content_length else {}

        candidate = body.get("candidate", "")
        example_id = body.get("example_id")

        if self.budget.exhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)
            return

        # Refuse example_ids outside train_set so val/test ids cannot be
        # probed via /evaluate. Single-task tasks have no examples, so
        # example_id is ignored there.
        if example_id is not None and self.task.has_dataset:
            if example_id not in self._train_ids():
                self._send_json(
                    handler,
                    {"error": f"Unknown or sealed example_id: {example_id!r}"},
                    status=400,
                )
                return

        try:
            example = self._examples.get(example_id) if example_id else None
            score, info = self.evaluate(candidate, example)
            self._send_json(handler, {"score": score, "info": info, "budget": self.budget.status()})

        except BudgetExhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)

        except ValueError as e:
            self._send_json(handler, {"error": str(e), "budget": self.budget.status()}, status=400)

        except Exception as e:
            self._send_json(handler, {"error": str(e), "budget": self.budget.status()}, status=500)

    def _handle_evaluate_examples(self, handler: BaseHTTPRequestHandler) -> None:
        content_length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(content_length)) if content_length else {}

        candidate = body.get("candidate", "")
        example_ids = body.get("example_ids")
        split = body.get("split", "train")

        if self.budget.exhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)
            return

        # Lock the HTTP surface to train only. val is exposed via /validate
        # (aggregate-only); test is not exposed at all. The Python API
        # (evaluate_examples) is unchanged so the runner can still score on
        # val/test directly for held-out reporting.
        if split not in ("train", None):
            self._send_json(
                handler,
                {"error": f"split={split!r} is not exposed via HTTP; only 'train' is allowed"},
                status=400,
            )
            return
        if example_ids is not None:
            train_ids = self._train_ids()
            invalid = [eid for eid in example_ids if eid not in train_ids]
            if invalid:
                self._send_json(
                    handler,
                    {"error": f"Unknown or sealed example_ids: {invalid[:5]}"},
                    status=400,
                )
                return

        try:
            avg_score, info = self.evaluate_examples(
                candidate,
                example_ids=example_ids,
                split="train" if example_ids is None else None,
            )
            self._send_json(handler, {
                "average_score": avg_score,
                "scores": info.get("scores", {}),
                "infos": info.get("infos", {}),
                "num_evaluated": info.get("num_evaluated", 1),
                "errors": info.get("errors", {}),
                "budget": self.budget.status(),
            })

        except BudgetExhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)

        except ValueError as e:
            self._send_json(handler, {"error": str(e), "budget": self.budget.status()}, status=400)

        except Exception as e:
            self._send_json(handler, {"error": str(e), "budget": self.budget.status()}, status=500)

    def _handle_validate(self, handler: BaseHTTPRequestHandler) -> None:
        content_length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(content_length)) if content_length else {}

        candidate = body.get("candidate", "")

        if not self.task.val_set:
            self._send_json(handler, {"error": "Task has no validation set"}, status=400)
            return

        if self.budget.exhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)
            return

        try:
            result = self.validate(candidate)
            self._send_json(handler, {**result, "budget": self.budget.status()})

        except BudgetExhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)

        except Exception as e:
            self._send_json(handler, {"error": str(e), "budget": self.budget.status()}, status=500)

    def _handle_status(self, handler: BaseHTTPRequestHandler) -> None:
        self._send_json(handler, {
            "budget": self.budget.status(),
            "task": self.task.name,
            "best_score": self.best_score,
            "best_visible_score": self.best_visible_score,
            "best_visible_source": self.best_visible_source,
            "best_validated_score": self.best_validated_score,
            "total_evals": self.budget.used,
            "total_cost": self.total_cost,
        })

    def _handle_task_info(self, handler: BaseHTTPRequestHandler) -> None:
        self._send_json(handler, {
            "name": self.task.name,
            "objective": self.task.objective,
            "background": self.task.background,
            "initial_candidate": self.task.initial_candidate,
            "has_dataset": self.task.has_dataset,
            "train_size": len(self.task.train_set) if self.task.train_set else 0,
            "val_size": len(self.task.val_set) if self.task.val_set else 0,
            "test_size": len(self.task.test_set) if self.task.test_set else 0,
            "metadata": {k: v for k, v in self.task.metadata.items() if isinstance(v, (str, int, float, bool))},
        })

    def _send_json(self, handler: BaseHTTPRequestHandler, data: dict, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
