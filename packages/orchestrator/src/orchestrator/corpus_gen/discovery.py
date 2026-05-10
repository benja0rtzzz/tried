"""Structural observation discovery over cloned Triton repos."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from shared.enums import OpCategory
from shared.logging import get_logger

from .patterns import (
    BroadcastPattern,
    DtypeMix,
    FusionShape,
    MemoryPattern,
    ReductionAxis,
    ShapeRank,
)

logger = get_logger(__name__)

DEFAULT_OUT = Path("data/corpus_gen/observations.jsonl")
DEFAULT_REPOS_ROOT = Path(".repos")
_SKIP_DIRS = {".git", "tests", "docs", "examples", "benchmarks", "__pycache__"}


def _attribute_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _is_triton_jit_decorator(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    path = _attribute_path(node)
    return path == "triton.jit"


def _read_repos_from_commits(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing commit manifest: {path}")

    repos: list[str] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            repos.append(parts[0])
    return repos


def _iter_python_files(repo_root: Path):
    for py_file in repo_root.rglob("*.py"):
        rel_parts = set(py_file.relative_to(repo_root).parts)
        if rel_parts & _SKIP_DIRS:
            continue
        yield py_file


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _classify_op_category(blob: str) -> OpCategory:
    if _contains_any(blob, ("softmax", "attn", "flash")):
        return OpCategory.FUSED_ATTENTION
    if _contains_any(blob, ("matmul", "gemm")):
        return OpCategory.MATMUL
    if _contains_any(blob, ("rmsnorm", "layer_norm", "layernorm", "norm")):
        return OpCategory.NORMALIZATION
    if _contains_any(blob, ("rope", "scan")):
        return OpCategory.OTHER
    if _contains_any(blob, ("conv",)):
        return OpCategory.CONVOLUTION
    if _contains_any(blob, ("embed", "emb")):
        return OpCategory.EMBEDDING
    if _contains_any(blob, ("loss", "ce", "jsd", "kl")):
        return OpCategory.LOSS
    if _contains_any(blob, ("quant", "dequant", "nf4", "fp8", "int4", "int8")):
        return OpCategory.QUANTIZATION
    if _contains_any(blob, ("relu", "gelu", "silu", "swiglu", "geglu")):
        return OpCategory.ACTIVATION
    if _contains_any(blob, ("sum", "mean", "max", "argmax")):
        return OpCategory.REDUCTION
    return OpCategory.ELEMENTWISE_CHAIN


def _classify_shape_rank(blob: str) -> ShapeRank:
    if _contains_any(blob, ("4d", "nchw", "nhwc", "conv2d", "image")):
        return ShapeRank.D4
    if _contains_any(blob, ("3d", "bth", "qkv", "sequence", "token")):
        return ShapeRank.D3
    if _contains_any(blob, ("1d", "vector", "flat")):
        return ShapeRank.D1
    return ShapeRank.D2


def _classify_dtype_mix(blob: str) -> DtypeMix:
    has_fp16 = "fp16" in blob or "float16" in blob
    has_bf16 = "bf16" in blob or "bfloat16" in blob
    has_fp32 = "fp32" in blob or "float32" in blob

    if "int8" in blob or "int4" in blob or "fp8" in blob:
        return DtypeMix.WITH_INT8
    if has_fp32 and has_fp16:
        return DtypeMix.MIXED_FP32_FP16
    if has_fp32 and has_bf16:
        return DtypeMix.MIXED_FP32_BF16
    if has_bf16:
        return DtypeMix.BF16_ONLY
    if has_fp16:
        return DtypeMix.FP16_ONLY
    return DtypeMix.FP32_ONLY


def _classify_broadcast(blob: str) -> BroadcastPattern:
    if "broadcast" not in blob:
        return BroadcastPattern.NONE
    if _contains_any(blob, ("channel", "channels")):
        return BroadcastPattern.CHANNEL
    if _contains_any(blob, ("batch", "bcast_b")):
        return BroadcastPattern.BATCH
    if _contains_any(blob, ("col", "column")):
        return BroadcastPattern.COL
    if _contains_any(blob, ("row",)):
        return BroadcastPattern.ROW
    return BroadcastPattern.NONE


def _classify_reduction_axis(blob: str) -> ReductionAxis:
    if not _contains_any(blob, ("reduce", "sum", "mean", "argmax", "amax", "max")):
        return ReductionAxis.NONE
    if _contains_any(blob, ("dim=-1", "axis=-1", "last")):
        return ReductionAxis.LAST
    if _contains_any(blob, ("all", "global")):
        return ReductionAxis.ALL
    if _contains_any(blob, ("channel", "dim=1", "axis=1")):
        return ReductionAxis.CHANNEL
    if _contains_any(blob, ("batch", "dim=0", "axis=0")):
        return ReductionAxis.BATCH
    return ReductionAxis.NONE


def _classify_fusion_shape(blob: str) -> FusionShape:
    if _contains_any(blob, ("reduce_then", "reduction_then", "sum_then", "mean_then")):
        return FusionShape.REDUCE_THEN_OP
    if _contains_any(blob, ("then_reduce", "op_then_reduce")):
        return FusionShape.OP_THEN_REDUCE

    op_markers = (
        "relu",
        "gelu",
        "silu",
        "swiglu",
        "geglu",
        "softmax",
        "matmul",
        "add",
        "mul",
        "sum",
        "mean",
        "norm",
    )
    score = sum(1 for marker in op_markers if marker in blob)
    if score >= 3:
        return FusionShape.TRIPLET
    if score >= 2:
        return FusionShape.PAIR
    return FusionShape.SINGLE_OP


def _classify_memory_pattern(blob: str) -> MemoryPattern:
    if _contains_any(blob, ("jagged", "ragged")):
        return MemoryPattern.JAGGED
    if _contains_any(blob, ("mask", "causal", "block_sparse")):
        return MemoryPattern.MASKED
    if _contains_any(blob, ("stride", "strided")):
        return MemoryPattern.STRIDED
    return MemoryPattern.CONTIGUOUS


def _load_seen(path: Path) -> set[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    if not path.exists():
        return seen

    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            repo = row.get("repo")
            file_path = row.get("file_path")
            function_name = row.get("function_name")
            if isinstance(repo, str) and isinstance(file_path, str) and isinstance(function_name, str):
                seen.add((repo, file_path, function_name))
    return seen


def run(out_path: Path, repos_root: Path) -> None:
    commits_path = repos_root / "COMMITS.txt"
    repo_names = _read_repos_from_commits(commits_path)
    seen = _load_seen(out_path)

    logger.info("loaded %d repos from %s", len(repo_names), commits_path)
    if seen:
        logger.info("resume: loaded %d existing observations", len(seen))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0

    with out_path.open("a") as out_file:
        for repo in repo_names:
            repo_root = repos_root / repo
            if not repo_root.exists():
                logger.warning("missing repo checkout, skipping: %s", repo_root)
                continue

            repo_added = 0
            for py_file in _iter_python_files(repo_root):
                rel_path = py_file.relative_to(repo_root).as_posix()
                try:
                    source = py_file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    source = py_file.read_text(encoding="utf-8", errors="ignore")

                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    continue

                for node in ast.walk(tree):
                    if not isinstance(node, ast.FunctionDef):
                        continue
                    if not any(_is_triton_jit_decorator(d) for d in node.decorator_list):
                        continue

                    key = (repo, rel_path, node.name)
                    if key in seen:
                        continue

                    blob = f"{rel_path} {node.name}".lower()

                    row = {
                        "repo": repo,
                        "file_path": rel_path,
                        "function_name": node.name,
                        "op_category": _classify_op_category(blob).value,
                        "shape_rank": _classify_shape_rank(blob).value,
                        "dtype_mix": _classify_dtype_mix(blob).value,
                        "broadcast_pattern": _classify_broadcast(blob).value,
                        "reduction_axis": _classify_reduction_axis(blob).value,
                        "fusion_shape": _classify_fusion_shape(blob).value,
                        "memory_pattern": _classify_memory_pattern(blob).value,
                    }
                    out_file.write(json.dumps(row) + "\n")
                    out_file.flush()
                    seen.add(key)
                    added += 1
                    repo_added += 1

            logger.info("repo %s: +%d new observations", repo, repo_added)

    logger.info("discovery complete: +%d new rows written to %s", added, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="orchestrator.corpus_gen.discovery")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repos-root", type=Path, default=DEFAULT_REPOS_ROOT)
    args = parser.parse_args()
    run(args.out, args.repos_root)


if __name__ == "__main__":
    main()
