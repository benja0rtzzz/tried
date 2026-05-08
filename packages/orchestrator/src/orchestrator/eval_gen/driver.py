"""End-to-end driver for stages 1-3 of the eval-set generation pipeline.

Reads specs.jsonl produced by sampler.main() and, for each spec:
  1. Renders the locked stage-2 prompt.
  2. Calls `codex exec` (unless --dry-run).
  3. Parses the response.
  4. Runs the AST validator from ast_check.validate.
  5. Runs the canonical-hash dedup against the training corpus and the
     intra-eval set.
  6. Writes accepted (spec, code) rows to with_code.jsonl; rejections
     (with reason) to rejected.jsonl.

Resumable: on restart, specs whose spec_id already appears in
with_code.jsonl OR rejected.jsonl are skipped. Output writes are
flushed per row so a crash leaves a recoverable state.

Note: stage 4 (`/preflight`) is NOT part of this driver. Block D will
add a separate driver that consumes with_code.jsonl and runs preflight
to produce the final `eval/holdout/synthetic_fusions.jsonl`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from shared.logging import get_logger
from shared.models import EvalSpec

from .ast_check import validate as ast_validate
from .codex import (
    CodexCallError,
    GeneratedCode,
    ParseError,
    UnrealizableSpec,
    synthesize,
)
from .dedup import IntraEvalDedup, canonical_hash, load_training_hashes
from .prompt import render as render_prompt

logger = get_logger(__name__)

DEFAULT_SPECS = Path("data/eval_gen/specs.jsonl")
DEFAULT_OUT = Path("data/eval_gen/with_code.jsonl")
DEFAULT_REJECTED = Path("data/eval_gen/rejected.jsonl")


# ---------------------------------------------------------------------------
# Output row shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WithCodeRow:
    spec: EvalSpec
    pytorch_code: str

    def to_json_line(self) -> str:
        return json.dumps({
            "spec": self.spec.model_dump(mode="json"),
            "pytorch_code": self.pytorch_code,
        }) + "\n"


@dataclass(frozen=True)
class RejectedRow:
    spec_id: str
    stage: str  # "codex" | "parse" | "ast" | "dedup_train" | "dedup_intra"
    reason: str
    where: str = ""

    def to_json_line(self) -> str:
        return json.dumps({
            "spec_id": self.spec_id,
            "stage": self.stage,
            "reason": self.reason,
            "where": self.where,
        }) + "\n"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_specs(path: Path) -> list[EvalSpec]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(EvalSpec.model_validate_json(line))
    return out


def collect_processed_spec_ids(*paths: Path) -> set[str]:
    """Read existing output files and return the set of spec_ids that
    already have a verdict. Used for resume."""
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "spec" in row:
                    seen.add(row["spec"]["spec_id"])
                elif "spec_id" in row:
                    seen.add(row["spec_id"])
    return seen


def append_jsonl(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line)
        f.flush()


# ---------------------------------------------------------------------------
# Per-spec pipeline
# ---------------------------------------------------------------------------

def process_one(
    spec: EvalSpec,
    intra: IntraEvalDedup,
    training_hashes: set[str],
    dry_run: bool,
) -> WithCodeRow | RejectedRow:
    prompt = render_prompt(spec)

    if dry_run:
        # Just emit the prompt; no Codex call, no validators run.
        print(f"=== spec_id={spec.spec_id} ({spec.form}) ===")
        print(prompt)
        print()
        return RejectedRow(
            spec_id=spec.spec_id, stage="dry_run", reason="dry-run mode",
        )

    try:
        result = synthesize(prompt)
    except CodexCallError as e:
        return RejectedRow(spec_id=spec.spec_id, stage="codex", reason=str(e))

    if isinstance(result, ParseError):
        return RejectedRow(spec_id=spec.spec_id, stage="parse", reason=result.detail)
    if isinstance(result, UnrealizableSpec):
        return RejectedRow(
            spec_id=spec.spec_id, stage="parse",
            reason=f"unrealizable: {result.explanation}",
        )
    assert isinstance(result, GeneratedCode)
    code = result.code

    ast_res = ast_validate(code, spec)
    if not ast_res.ok:
        return RejectedRow(
            spec_id=spec.spec_id, stage="ast",
            reason=ast_res.reason, where=ast_res.where,
        )

    h = canonical_hash(code)
    if h in training_hashes:
        return RejectedRow(
            spec_id=spec.spec_id, stage="dedup_train",
            reason="canonical-AST hash collides with training corpus row",
        )
    intra_collision = intra.check(code)
    if intra_collision is not None:
        return RejectedRow(
            spec_id=spec.spec_id, stage="dedup_intra",
            reason=f"canonical-AST hash collides with already-accepted eval row ({intra_collision[:12]}...)",
        )

    return WithCodeRow(spec=spec, pytorch_code=code)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    specs_path: Path,
    out_path: Path,
    rejected_path: Path,
    dry_run: bool,
    limit: int | None,
) -> None:
    specs = load_specs(specs_path)
    logger.info(f"loaded {len(specs)} specs from {specs_path}")

    if limit is not None:
        specs = specs[:limit]
        logger.info(f"limit={limit} applied")

    processed = collect_processed_spec_ids(out_path, rejected_path)
    if processed:
        before = len(specs)
        specs = [s for s in specs if s.spec_id not in processed]
        logger.info(f"resume: skipping {before - len(specs)} already-processed specs")

    if not dry_run:
        training_hashes = load_training_hashes()
        logger.info(f"loaded {len(training_hashes)} training-corpus hashes")
    else:
        training_hashes = set()

    intra = IntraEvalDedup()

    n_accepted = n_rejected = 0
    for i, spec in enumerate(specs):
        row = process_one(spec, intra, training_hashes, dry_run=dry_run)
        if isinstance(row, WithCodeRow):
            append_jsonl(out_path, row.to_json_line())
            n_accepted += 1
        else:
            append_jsonl(rejected_path, row.to_json_line())
            n_rejected += 1
            if not dry_run:
                logger.info(
                    f"reject [{i+1}/{len(specs)}] {spec.spec_id[:8]} "
                    f"stage={row.stage} reason={row.reason}"
                )
        if (i + 1) % 25 == 0:
            logger.info(f"progress: {i+1}/{len(specs)} accepted={n_accepted} rejected={n_rejected}")

    logger.info(f"done: accepted={n_accepted} rejected={n_rejected} total={len(specs)}")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="orchestrator.eval_gen.driver",
        description="Run stage 2 (Codex) + stage 3 (AST + dedup) on sampled specs.",
    )
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print rendered prompts; do not call Codex or run validators",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="process only the first N specs (after resume filtering)",
    )
    args = parser.parse_args()

    run(args.specs, args.out, args.rejected, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
