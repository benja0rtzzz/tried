"""Synthesis driver: spec -> codex -> AST check -> eval dedup."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from shared.enums import Dtype, OpCategory, TolerancePolicy
from shared.logging import get_logger
from shared.models import PreflightSafeRecord

from .ast_check import ValidationError as ASTValidationError
from .ast_check import validate as ast_validate
from .codex import CodexCallError, ParseError, RateLimitError, synthesize
from .dedup import EvalDedup, derive_example_id
from .patterns import SkeletonSpec

logger = get_logger(__name__)
DEFAULT_SPECS = Path("data/corpus_gen/specs.jsonl")
DEFAULT_OUT = Path("data/corpus_gen/with_code.jsonl")
DEFAULT_REJECTED = Path("data/corpus_gen/rejected.jsonl")
DEFAULT_EVAL = Path("eval/holdout/synthetic_fusions.jsonl")


@dataclass(frozen=True)
class WithCodeRow:
    spec_id: str
    spec: SkeletonSpec
    candidate: PreflightSafeRecord
    rationale: str

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "spec_id": self.spec_id,
                "spec": self.spec.model_dump(mode="json"),
                "candidate": self.candidate.model_dump(mode="json"),
                "rationale": self.rationale,
            }
        ) + "\n"


@dataclass(frozen=True)
class RejectedRow:
    spec_id: str
    reason: str
    message: str

    def to_json_line(self) -> str:
        return json.dumps({"spec_id": self.spec_id, "reason": self.reason, "message": self.message}) + "\n"


def _load_specs(path: Path) -> list[SkeletonSpec]:
    out: list[SkeletonSpec] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if line:
                out.append(SkeletonSpec.model_validate_json(line))
    return out


def _collect_processed_spec_ids(*paths: Path) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                spec_id = row.get("spec_id")
                if isinstance(spec_id, str):
                    seen.add(spec_id)
                    continue
                spec = row.get("spec")
                if isinstance(spec, dict) and isinstance(spec.get("spec_id"), str):
                    seen.add(spec["spec_id"])
    return seen


_WRITE_LOCK = threading.Lock()


def _append_jsonl(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a") as f:
        f.write(line)
        f.flush()


def _process_one(spec: SkeletonSpec, dedup: EvalDedup) -> tuple[str, WithCodeRow | RejectedRow | str]:
    try:
        response = synthesize(spec)
    except RateLimitError as exc:
        return "rate_limit", str(exc)
    except CodexCallError as exc:
        return "rejected", RejectedRow(spec.spec_id, "codex_call", str(exc))
    except ParseError as exc:
        return "rejected", RejectedRow(spec.spec_id, "parse", str(exc))

    try:
        ast_validate(response.pytorch_code)
    except ASTValidationError as exc:
        return "rejected", RejectedRow(spec.spec_id, "ast", str(exc))

    example_id = derive_example_id(response.pytorch_code)
    collision = dedup.is_collision(response.pytorch_code, example_id)
    if collision is not None:
        return "rejected", RejectedRow(spec.spec_id, "dedup_eval", collision)

    try:
        candidate = PreflightSafeRecord(
            example_id=example_id,
            op_category=spec.op_category,
            pytorch_code=response.pytorch_code,
            input_shapes=response.input_shapes,
            input_dtypes=response.input_dtypes,
            rng_seed=spec.rng_seed,
            tolerance_policy=_tolerance_policy_for(
                spec.op_category,
                response.input_dtypes,
            ),
        )
    except ValidationError as exc:
        return "rejected", RejectedRow(spec.spec_id, "candidate_schema", str(exc))

    return "success", WithCodeRow(spec.spec_id, spec, candidate, response.rationale)


def _tolerance_policy_for(
    op_category: OpCategory,
    input_dtypes: list[Dtype],
) -> TolerancePolicy:
    if all(dtype in {Dtype.INT8, Dtype.INT16, Dtype.INT32, Dtype.INT64, Dtype.BOOL} for dtype in input_dtypes):
        return TolerancePolicy.EXACT_INTEGER
    if any(dtype in {Dtype.FLOAT16, Dtype.BFLOAT16} for dtype in input_dtypes):
        if op_category == OpCategory.REDUCTION:
            return TolerancePolicy.REDUCTION_FP16
        if op_category == OpCategory.FUSED_ATTENTION:
            return TolerancePolicy.ATTENTION_SOFTMAX_FP16
        return TolerancePolicy.DEFAULT_FP16
    if op_category == OpCategory.REDUCTION:
        return TolerancePolicy.REDUCTION_FP32
    return TolerancePolicy.DEFAULT_FP32


def _record_result(result: tuple[str, WithCodeRow | RejectedRow | str], out_path: Path, rejected_path: Path) -> tuple[bool, int, int]:
    kind, payload = result
    if kind == "rate_limit":
        return True, 0, 0
    if kind == "success":
        _append_jsonl(out_path, payload.to_json_line())
        return False, 1, 0
    _append_jsonl(rejected_path, payload.to_json_line())
    return False, 0, 1


def _run_sequential(specs: list[SkeletonSpec], dedup: EvalDedup, out_path: Path, rejected_path: Path) -> bool:
    accepted = 0
    rejected = 0
    for i, spec in enumerate(specs, start=1):
        result = _process_one(spec, dedup)
        hit_rate, inc_acc, inc_rej = _record_result(result, out_path, rejected_path)
        accepted += inc_acc
        rejected += inc_rej

        if result[0] == "rejected":
            row = result[1]
            logger.info("reject [%d/%d] %s reason=%s message=%s", i, len(specs), spec.spec_id[:8], row.reason, row.message)
        if i % 25 == 0:
            logger.info("progress: %d/%d accepted=%d rejected=%d", i, len(specs), accepted, rejected)
        if hit_rate:
            logger.error("codex rate limit encountered after %d rows", i - 1)
            logger.info("partial progress: accepted=%d rejected=%d", accepted, rejected)
            return True

    logger.info("done: accepted=%d rejected=%d total=%d", accepted, rejected, len(specs))
    return False


def _run_parallel(specs: list[SkeletonSpec], dedup: EvalDedup, out_path: Path, rejected_path: Path, parallel: int) -> bool:
    accepted = 0
    rejected = 0
    completed = 0
    hit_rate_limit = False

    executor = ThreadPoolExecutor(max_workers=parallel)
    pending: dict[Future[tuple[str, WithCodeRow | RejectedRow | str]], SkeletonSpec] = {}
    spec_iter = iter(specs)

    def submit_next() -> bool:
        try:
            spec = next(spec_iter)
        except StopIteration:
            return False
        pending[executor.submit(_process_one, spec, dedup)] = spec
        return True

    for _ in range(min(parallel, len(specs))):
        submit_next()

    while pending and not hit_rate_limit:
        done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
        for future in done:
            spec = pending.pop(future)
            result = future.result()
            hit_rate, inc_acc, inc_rej = _record_result(result, out_path, rejected_path)

            completed += 1
            accepted += inc_acc
            rejected += inc_rej

            if result[0] == "rejected":
                row = result[1]
                logger.info("reject [%d/%d] %s reason=%s message=%s", completed, len(specs), spec.spec_id[:8], row.reason, row.message)
            if completed % 25 == 0:
                logger.info("progress: %d/%d accepted=%d rejected=%d", completed, len(specs), accepted, rejected)
            if hit_rate:
                logger.error("codex rate limit encountered after %d rows", completed - 1)
                logger.info("partial progress: accepted=%d rejected=%d", accepted, rejected)
                hit_rate_limit = True
                break

            submit_next()

    if hit_rate_limit:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return True

    executor.shutdown(wait=True)
    logger.info("done: accepted=%d rejected=%d total=%d", accepted, rejected, len(specs))
    return False


def run(specs_path: Path, out_path: Path, rejected_path: Path, eval_path: Path, parallel: int) -> bool:
    specs = _load_specs(specs_path)
    logger.info("loaded %d specs from %s", len(specs), specs_path)

    processed = _collect_processed_spec_ids(out_path, rejected_path)
    if processed:
        before = len(specs)
        specs = [spec for spec in specs if spec.spec_id not in processed]
        logger.info("resume: skipping %d already-processed specs", before - len(specs))
    logger.info("specs to process this run: %d", len(specs))

    dedup = EvalDedup(eval_path)
    return _run_sequential(specs, dedup, out_path, rejected_path) if parallel <= 1 else _run_parallel(specs, dedup, out_path, rejected_path, parallel)


def main() -> None:
    parser = argparse.ArgumentParser(prog="orchestrator.corpus_gen.driver")
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    if run(
        specs_path=args.specs,
        out_path=args.out,
        rejected_path=args.rejected,
        eval_path=args.eval,
        parallel=max(1, args.parallel),
    ):
        sys.exit(0)


if __name__ == "__main__":
    main()
