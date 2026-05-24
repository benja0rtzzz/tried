"""Run /preflight on synthesized corpus candidates and append slim survivors.

This module is kept for compatibility with the older corpus_gen entry point.
The active Step 5 preflight command is orchestrator.train.dataset.preflight_driver;
both write PreflightSafeRecord rows to data/preflight_safe.jsonl.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from orchestrator.clients.verification_client import make_client
from shared.enums import OpCategory
from shared.logging import get_logger
from shared.models import PreflightSafeRecord
from shared.verification.api import PreflightRequest, PreflightResponse

logger = get_logger(__name__)

DEFAULT_WITH_CODE = Path("data/corpus_gen/with_code.jsonl")
DEFAULT_OUT = Path("data/preflight_safe.jsonl")
DEFAULT_SKIPPED = Path("data/preflight_rejected.jsonl")


def _parse_allowed_ops(raw_values: list[str] | None) -> set[OpCategory] | None:
    if raw_values is None:
        return None

    values = [
        value.strip()
        for raw in raw_values
        for value in raw.split(",")
        if value.strip()
    ]
    valid = {category.value for category in OpCategory}
    invalid = [value for value in values if value not in valid]
    if invalid:
        raise ValueError(
            "invalid op categories: "
            + ", ".join(invalid)
            + "; valid values: "
            + ", ".join(sorted(valid))
        )
    if not values:
        raise ValueError("--allowed-ops was provided but no categories were listed")
    return {OpCategory(value) for value in values}


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


def _build_record(row: _WithCodeRow) -> PreflightSafeRecord:
    return row.candidate


def run(
    with_code_path: Path,
    out_path: Path,
    skipped_path: Path,
    allowed_ops: set[OpCategory] | None = None,
) -> None:
    rows = _load_with_code(with_code_path)
    logger.info("loaded %d candidate rows from %s", len(rows), with_code_path)

    if allowed_ops is not None:
        before = len(rows)
        rows = [row for row in rows if row.candidate.op_category in allowed_ops]
        logger.info(
            "allowed ops %s: filtered %d candidate row(s), %d remain",
            [category.value for category in sorted(allowed_ops, key=lambda c: c.value)],
            before - len(rows),
            len(rows),
        )

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
        policy = record.tolerance_policy

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
    parser = argparse.ArgumentParser(prog="orchestrator.train.corpus_gen.preflight_driver")
    parser.add_argument("--with-code", type=Path, default=DEFAULT_WITH_CODE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skipped", type=Path, default=DEFAULT_SKIPPED)
    parser.add_argument(
        "--allowed-ops",
        nargs="+",
        default=None,
        metavar="OP_CATEGORY",
        help="only preflight rows from these op categories; accepts spaces or commas",
    )
    args = parser.parse_args()
    try:
        allowed_ops = _parse_allowed_ops(args.allowed_ops)
    except ValueError as exc:
        parser.error(str(exc))
    run(args.with_code, args.out, args.skipped, allowed_ops)


if __name__ == "__main__":
    main()
