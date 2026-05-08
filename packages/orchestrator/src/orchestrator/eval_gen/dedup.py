"""Stage 3b — Canonical-AST dedup against the training corpus and within
the eval set.

Catches the case where the LLM produces pytorch_code that is structurally
identical (modulo identifier names and whitespace) to a row in the
training corpus or to another already-accepted eval row.

Limitations (acceptable for v1):
  - Does not catch semantically equivalent code that uses different ops
    (e.g. relu(x) vs (x > 0) * x — but operator-form is rejected by
    stage-3a anyway).
  - Does not catch arg-order swaps in commutative ops (torch.add(a, b)
    vs torch.add(b, a)).
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from shared.dataset import load_corpus_train


# Names that should NOT be renamed during canonicalization. Module names,
# the function name itself (we rename it explicitly to "candidate"), and
# constant builtins.
_PRESERVED_NAMES: frozenset[str] = frozenset({
    "torch", "True", "False", "None",
})


class _IdNormalizer(ast.NodeTransformer):
    """Rename every local Name / FunctionDef arg / Assign target to
    v0, v1, ... in order of first appearance. Function name normalized
    to 'candidate'."""

    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self.counter = 0

    def _rename(self, name: str) -> str:
        if name in _PRESERVED_NAMES:
            return name
        if name not in self.mapping:
            self.mapping[name] = f"v{self.counter}"
            self.counter += 1
        return self.mapping[name]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = "candidate"
        for arg in node.args.args:
            arg.arg = self._rename(arg.arg)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._rename(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._rename(node.arg)
        return node


def canonical_hash(pytorch_code: str) -> str:
    """Return a stable SHA-256 of the canonicalized AST. Two pieces of
    code that differ only in identifier names, whitespace, or comments
    produce the same hash."""
    try:
        module = ast.parse(pytorch_code)
    except SyntaxError:
        # Unparseable code is its own bucket.
        return hashlib.sha256(b"__unparseable__" + pytorch_code.encode()).hexdigest()
    norm = _IdNormalizer().visit(module)
    ast.fix_missing_locations(norm)
    canonical = ast.unparse(norm)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Training-corpus hash cache
# ---------------------------------------------------------------------------

_TRAIN_HASH_CACHE: set[str] | None = None


def load_training_hashes(corpus_path: Path | str = "data/corpus_train.jsonl") -> set[str]:
    """Compute (and cache) canonical hashes of every training-corpus row's
    pytorch_code. Cheap per-row; full pass on first call only."""
    global _TRAIN_HASH_CACHE
    if _TRAIN_HASH_CACHE is None:
        rows = load_corpus_train(corpus_path)
        _TRAIN_HASH_CACHE = {canonical_hash(r.pytorch_code) for r in rows}
    return _TRAIN_HASH_CACHE


def is_training_dupe(pytorch_code: str, training_hashes: set[str] | None = None) -> bool:
    if training_hashes is None:
        training_hashes = load_training_hashes()
    return canonical_hash(pytorch_code) in training_hashes


# ---------------------------------------------------------------------------
# Intra-eval dedup state
# ---------------------------------------------------------------------------

class IntraEvalDedup:
    """Tracks accepted-so-far eval candidate hashes; rejects collisions."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def check(self, pytorch_code: str) -> str | None:
        """Returns None if no collision (and registers the hash);
        returns the colliding hash string if duplicate."""
        h = canonical_hash(pytorch_code)
        if h in self._seen:
            return h
        self._seen.add(h)
        return None

    @property
    def n_seen(self) -> int:
        return len(self._seen)
