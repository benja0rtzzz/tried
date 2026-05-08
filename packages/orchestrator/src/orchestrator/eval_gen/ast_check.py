"""Stage 3a — Static AST guardrails for candidate pytorch_code.

Validates a string of Python source against an EvalSpec. Used between
stage 2 (Codex code synthesis) and stage 4 (/preflight): rejections here
are deterministic and free, so we filter aggressively before paying for
verification-server compile cycles.

Checks (in order, first failure wins):
  1.  Parses as Python.
  2.  Module body contains only `import torch` and a single function def.
  3.  No `from ... import ...`, no aliased imports.
  4.  Function takes exactly len(spec.input_shapes) positional args.
  5.  No torch.rand* / torch.empty* / torch.zeros* / torch.ones* / .to()
      / .cuda() / .cpu() (non-deterministic or device-changing).
  6.  Function body uses operator-form binary or unary ops (e.g. x + y,
      -x) — REJECTED. Function-form (torch.add) only.
  7.  Every Call whose function is in shared.eval.base_ops.BASE_OPS is
      counted; the resulting op sequence must exactly equal spec.ops.
  8.  Any Call to a name outside BASE_OPS ∪ FREE_MOVEMENT is rejected.
  9.  No nested function defs, classes, comprehensions, or control flow
      (eval candidates are straight-line tensor pipelines).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

from shared.eval.base_ops import BASE_OPS, FREE_MOVEMENT
from shared.eval.forms import FORMS
from shared.models import EvalSpec


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = ""
    where: str = ""  # short location hint, e.g. "line 7"

    @classmethod
    def fail(cls, reason: str, where: str = "") -> "ValidationResult":
        return cls(ok=False, reason=reason, where=where)


_OK = ValidationResult(ok=True)


# Forbidden top-level paths. Stage 3 doesn't try to be exhaustive — these
# are the patterns the LLM is most likely to produce that would silently
# break determinism or leave the verification server.
_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "torch.rand", "torch.empty", "torch.zeros", "torch.ones",
    "torch.full", "torch.tensor",  # explicit value injection, also forbidden
    "torch.cuda", "torch.cpu",
)
_FORBIDDEN_TENSOR_METHODS: frozenset[str] = frozenset({
    "to", "cuda", "cpu", "type", "type_as",
    "fill_", "zero_", "uniform_", "normal_", "random_",
})


def _attribute_path(node: ast.AST) -> str | None:
    """Reconstruct dotted access like `torch.nn.functional.gelu`. Returns
    None if not a clean attribute chain."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _validate_imports(module: ast.Module) -> ValidationResult:
    """Walk module-level statements; require exactly `import torch` and
    one function def."""
    func_seen = False
    torch_import_seen = False
    for node in module.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "torch" or alias.asname is not None:
                    return ValidationResult.fail(
                        f"forbidden import: {alias.name}"
                        f"{f' as {alias.asname}' if alias.asname else ''}",
                        where=f"line {node.lineno}",
                    )
                torch_import_seen = True
        elif isinstance(node, ast.FunctionDef):
            if func_seen:
                return ValidationResult.fail(
                    "multiple function definitions",
                    where=f"line {node.lineno}",
                )
            func_seen = True
        elif isinstance(node, (ast.ImportFrom, ast.ClassDef, ast.Assign,
                                ast.AnnAssign, ast.AsyncFunctionDef)):
            return ValidationResult.fail(
                f"forbidden module-level statement: {type(node).__name__}",
                where=f"line {node.lineno}",
            )
        else:
            return ValidationResult.fail(
                f"forbidden module-level statement: {type(node).__name__}",
                where=f"line {node.lineno}",
            )
    if not torch_import_seen:
        return ValidationResult.fail("missing required `import torch` at module level")
    if not func_seen:
        return ValidationResult.fail("no function definition")
    return _OK


def _get_function(module: ast.Module) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise AssertionError("unreachable — _validate_imports should have caught this")


def _validate_signature(func: ast.FunctionDef, n_inputs: int) -> ValidationResult:
    args = func.args
    if (args.vararg is not None or args.kwarg is not None
            or args.kwonlyargs or args.kw_defaults or args.defaults
            or args.posonlyargs):
        return ValidationResult.fail(
            "function signature must be plain positional args, no defaults / *args / **kwargs",
            where=f"line {func.lineno}",
        )
    if len(args.args) != n_inputs:
        return ValidationResult.fail(
            f"function takes {len(args.args)} args; spec has {n_inputs} inputs",
            where=f"line {func.lineno}",
        )
    return _OK


def _walk_body_collect_ops(func: ast.FunctionDef) -> tuple[list[str], ValidationResult]:
    """Walk every Call node inside the function body in source order.
    Returns (ordered_ops, result_or_OK). ordered_ops contains only names
    that appear in BASE_OPS, in the order their Calls execute (post-order
    of the AST = innermost first, which matches Python evaluation order).
    """
    ops_in_order: list[str] = []
    failure: ValidationResult | None = None

    class _Visitor(ast.NodeVisitor):
        def visit_BinOp(self, node: ast.BinOp) -> None:
            nonlocal failure
            if failure is None:
                failure = ValidationResult.fail(
                    f"operator-form binary op ({type(node.op).__name__}); use function-form (torch.add etc.)",
                    where=f"line {node.lineno}",
                )

        def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
            nonlocal failure
            if failure is not None or isinstance(node.op, ast.Not):
                self.generic_visit(node)
                return
            # Allow USub / UAdd on numeric literals (e.g. dim=-1, scale=-0.5).
            # Reject when operand isn't a literal — that's the tensor case.
            if (isinstance(node.op, (ast.USub, ast.UAdd))
                    and isinstance(node.operand, ast.Constant)
                    and isinstance(node.operand.value, (int, float))):
                return
            failure = ValidationResult.fail(
                f"operator-form unary op ({type(node.op).__name__}) on non-literal operand; use function-form",
                where=f"line {node.lineno}",
            )

        def visit_For(self, node: ast.For) -> None:
            nonlocal failure
            if failure is None:
                failure = ValidationResult.fail("control flow not allowed (for)", where=f"line {node.lineno}")

        def visit_While(self, node: ast.While) -> None:
            nonlocal failure
            if failure is None:
                failure = ValidationResult.fail("control flow not allowed (while)", where=f"line {node.lineno}")

        def visit_If(self, node: ast.If) -> None:
            nonlocal failure
            if failure is None:
                failure = ValidationResult.fail("control flow not allowed (if)", where=f"line {node.lineno}")

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            nonlocal failure
            if failure is None:
                failure = ValidationResult.fail("nested function def not allowed", where=f"line {node.lineno}")

        def visit_Lambda(self, node: ast.Lambda) -> None:
            nonlocal failure
            if failure is None:
                failure = ValidationResult.fail("lambda not allowed", where=f"line {node.lineno}")

        def _flag_comp(
            self,
            node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        ) -> None:
            nonlocal failure
            if failure is None:
                failure = ValidationResult.fail("comprehension not allowed", where=f"line {node.lineno}")

        def visit_ListComp(self, node: ast.ListComp) -> None:
            self._flag_comp(node)

        def visit_SetComp(self, node: ast.SetComp) -> None:
            self._flag_comp(node)

        def visit_DictComp(self, node: ast.DictComp) -> None:
            self._flag_comp(node)

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            self._flag_comp(node)

        def visit_Call(self, node: ast.Call) -> None:
            nonlocal failure
            # Visit args first so inner calls register first (matches
            # actual Python eval order: innermost first).
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw.value)

            path = _attribute_path(node.func)

            if path is None:
                # The Call's func is something more complex than a clean
                # attribute chain rooted at a Name (e.g. method on a Call
                # result, subscript, etc.). Reject — only simple forms
                # allowed.
                if failure is None:
                    failure = ValidationResult.fail(
                        "complex call expression; use only torch.X(...) or x.method(...)",
                        where=f"line {node.lineno}",
                    )
                return

            if path in BASE_OPS:
                ops_in_order.append(path)
                return
            if path in FREE_MOVEMENT:
                return

            if path.startswith("torch."):
                # torch.X call but not in BASE_OPS / FREE_MOVEMENT — reject.
                if any(path.startswith(p) for p in _FORBIDDEN_PREFIXES):
                    if failure is None:
                        failure = ValidationResult.fail(
                            f"forbidden op: {path}", where=f"line {node.lineno}",
                        )
                    return
                if failure is None:
                    failure = ValidationResult.fail(
                        f"op outside BASE_OPS allow-list: {path}",
                        where=f"line {node.lineno}",
                    )
                return

            # Tensor method call: x.method(...). Path looks like "x.method"
            # or "x.y.method"; the method name is the last component.
            method = path.rsplit(".", 1)[-1]
            if method in FREE_MOVEMENT:
                return
            if method in _FORBIDDEN_TENSOR_METHODS:
                if failure is None:
                    failure = ValidationResult.fail(
                        f"forbidden tensor method: .{method}()", where=f"line {node.lineno}",
                    )
                return
            if failure is None:
                failure = ValidationResult.fail(
                    f"tensor method outside FREE_MOVEMENT: .{method}()",
                    where=f"line {node.lineno}",
                )

    visitor = _Visitor()
    for stmt in func.body:
        visitor.visit(stmt)
    return ops_in_order, (failure if failure is not None else _OK)


def validate(pytorch_code: str, spec: EvalSpec) -> ValidationResult:
    """Top-level: parse + run all guards. Returns ValidationResult."""
    try:
        module = ast.parse(pytorch_code)
    except SyntaxError as e:
        return ValidationResult.fail(f"syntax error: {e.msg}", where=f"line {e.lineno}")

    res = _validate_imports(module)
    if not res.ok:
        return res

    func = _get_function(module)
    res = _validate_signature(func, n_inputs=len(spec.input_shapes))
    if not res.ok:
        return res

    ops_seen, res = _walk_body_collect_ops(func)
    if not res.ok:
        return res

    expected_op_count = FORMS[spec.form].op_count
    if len(ops_seen) != expected_op_count:
        return ValidationResult.fail(
            f"op count mismatch: AST has {len(ops_seen)} BASE_OPS calls; "
            f"spec.form requires {expected_op_count}",
        )

    if tuple(ops_seen) != tuple(spec.ops):
        return ValidationResult.fail(
            f"op order mismatch: AST = {ops_seen}; spec.ops = {spec.ops}",
        )

    return _OK
