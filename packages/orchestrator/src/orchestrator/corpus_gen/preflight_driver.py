"""Run /preflight on synthesized corpus candidates and append survivors."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from orchestrator.clients.verification_client import make_client
from shared.enums import Dtype, OpCategory, TolerancePolicy
from shared.logging import get_logger
from shared.models import CorpusRecord, PreflightSafeRecord
from shared.verification.api import PreflightRequest, PreflightResponse

logger = get_logger(__name__)

DEFAULT_WITH_CODE = Path("data/corpus_gen/with_code.jsonl")
DEFAULT_OUT = Path("data/corpus_train.jsonl")
DEFAULT_SKIPPED = Path("data/corpus_gen/preflight_skipped.jsonl")


class _WithCodeRow(BaseModel):
    spec_id: str
    candidate: PreflightSafeRecord
    rationale: str | None = None


@dataclass(frozen=True)
class _SkippedRow:
    example_id: str
    reason: str

    def to_json_line(self) -> str:
        return json.dumps({"example_id": self.example_id, "reason": self.reason}) + "\n"


def _load_with_code(path: Path) -> list[_WithCodeRow]:
    rows: list[_WithCodeRow] = []
    with path.open() as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                rows.append(_WithCodeRow.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning("skipping malformed with_code row %s:%d (%s)", path, i, exc)
    return rows


def _collect_seen_example_ids(*paths: Path) -> set[str]:
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
                if isinstance(row, dict):
                    example_id = row.get("example_id")
                    if isinstance(example_id, str):
                        seen.add(example_id)
    return seen


def _append_jsonl(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line)
        f.flush()


def _select_policy(dtypes: list[Dtype], op_category: OpCategory) -> TolerancePolicy:
    fp16_dtypes = {Dtype.FLOAT16, Dtype.BFLOAT16}
    int_dtypes = {Dtype.INT8, Dtype.INT16, Dtype.INT32, Dtype.INT64, Dtype.BOOL}
    dtype_set = set(dtypes)

    if dtype_set <= int_dtypes:
        return TolerancePolicy.EXACT_INTEGER

    has_fp16 = bool(dtype_set & fp16_dtypes)

    if op_category == OpCategory.QUANTIZATION:
        return TolerancePolicy.LOW_PRECISION_DEQUANT
    if op_category == OpCategory.FUSED_ATTENTION and has_fp16:
        return TolerancePolicy.ATTENTION_SOFTMAX_FP16
    if op_category == OpCategory.REDUCTION:
        return TolerancePolicy.REDUCTION_FP16 if has_fp16 else TolerancePolicy.REDUCTION_FP32
    return TolerancePolicy.DEFAULT_FP16 if has_fp16 else TolerancePolicy.DEFAULT_FP32


def _classify_preflight(response: PreflightResponse) -> tuple[bool, str]:
    if response.error_message is not None:
        return False, f"harness error: {response.error_message}"
    if not response.passed:
        if response.vs_eager_inductor is not None:
            stats = response.vs_eager_inductor
            return False, (
                f"eager-vs-inductor disagreement: max_abs={stats.max_abs_diff:.3e} "
                f"max_rel={stats.max_rel_diff:.3e} "
                f"pct_exceeding={stats.pct_elements_exceeding_tol:.2f}"
            )
        return False, "preflight failed without diagnostics"
    return True, ""


def _build_record(row: _WithCodeRow) -> CorpusRecord:
    return row.candidate.to_corpus_record()


def run(with_code_path: Path, out_path: Path, skipped_path: Path) -> None:
    rows = _load_with_code(with_code_path)
    logger.info("loaded %d candidate rows from %s", len(rows), with_code_path)

    seen = _collect_seen_example_ids(out_path, skipped_path)
    if seen:
        before = len(rows)
        rows = [row for row in rows if row.candidate.example_id not in seen]
        logger.info("resume: skipping %d already-processed candidates", before - len(rows))

    client = make_client()

    accepted = 0
    skipped = 0
    for i, row in enumerate(rows, start=1):
        record = _build_record(row)
        policy = _select_policy(record.input_dtypes, record.op_category)

        try:
            response = client.preflight(
                PreflightRequest(
                    pytorch_code=record.pytorch_code,
                    input_shapes=record.input_shapes,
                    input_dtypes=record.input_dtypes,
                    rng_seed=record.rng_seed,
                    tolerance_policy=policy,
                )
            )
        except Exception as exc:
            _append_jsonl(
                skipped_path,
                _SkippedRow(
                    example_id=record.example_id,
                    reason=f"transport error: {type(exc).__name__}: {exc}",
                ).to_json_line(),
            )
            skipped += 1
            logger.error("transport error on %s: %s", record.example_id[:8], exc)
            continue

        passed, reason = _classify_preflight(response)
        if passed:
            _append_jsonl(out_path, record.model_dump_json() + "\n")
            accepted += 1
        else:
            _append_jsonl(
                skipped_path,
                _SkippedRow(example_id=record.example_id, reason=reason).to_json_line(),
            )
            skipped += 1
            logger.info("preflight skip %s: %s", record.example_id[:8], reason)

        if i % 10 == 0:
            logger.info(
                "progress: %d/%d accepted=%d skipped=%d",
                i,
                len(rows),
                accepted,
                skipped,
            )

    logger.info("done: accepted=%d skipped=%d total=%d", accepted, skipped, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(prog="orchestrator.corpus_gen.preflight_driver")
    parser.add_argument("--with-code", type=Path, default=DEFAULT_WITH_CODE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skipped", type=Path, default=DEFAULT_SKIPPED)
    args = parser.parse_args()
    run(args.with_code, args.out, args.skipped)


if __name__ == "__main__":
    main()
