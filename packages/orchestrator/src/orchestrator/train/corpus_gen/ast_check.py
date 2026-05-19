"""Static AST guardrails for generated PyTorch skeleton code."""
from __future__ import annotations

import ast

_BANNED_CTRL_FLOW = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)
_BANNED_EXPR_FORMS = (
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.Lambda,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)

_BANNED_TENSOR_METHODS = {
    "to",
    "cuda",
    "cpu",
    "type",
    "type_as",
    "float",
    "half",
    "double",
    "bfloat16",
}

_BANNED_TORCH_PREFIXES = (
    "torch.rand",
    "torch.empty",
    "torch.zeros",
    "torch.ones",
)
_BANNED_TORCH_EXACT = {
    "torch.full",
    "torch.tensor",
    "torch.arange",
}


class ValidationError(Exception):
    """Raised when candidate pytorch_code violates AST constraints."""


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


def _validate_module(module: ast.Module) -> ast.FunctionDef:
    if len(module.body) != 2:
        raise ValidationError("module body must be exactly: import torch, then one function")

    import_node, fn_node = module.body
    if not isinstance(import_node, ast.Import):
        raise ValidationError("first module statement must be `import torch`")
    if len(import_node.names) != 1:
        raise ValidationError("only `import torch` is allowed")
    alias = import_node.names[0]
    if alias.name != "torch" or alias.asname is not None:
        raise ValidationError("import must be exactly `import torch` with no alias")

    if not isinstance(fn_node, ast.FunctionDef):
        raise ValidationError("second module statement must be exactly one function definition")
    if fn_node.decorator_list:
        raise ValidationError("function decorators are not allowed")

    return fn_node


def _validate_signature(fn: ast.FunctionDef) -> None:
    args = fn.args
    n_positional = len(args.posonlyargs) + len(args.args)
    if n_positional < 1:
        raise ValidationError("function must take at least one positional argument")
    if args.vararg is not None or args.kwarg is not None or args.kwonlyargs:
        raise ValidationError("varargs, kwargs, and keyword-only args are not allowed")


class _GuardVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.return_count = 0
        self.error: str | None = None

    def _fail(self, message: str, node: ast.AST) -> None:
        if self.error is None:
            self.error = f"{message} (line {getattr(node, 'lineno', '?')})"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._fail("nested function definitions are not allowed", node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._fail("async functions are not allowed", node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._fail("class definitions are not allowed", node)

    def visit_Return(self, node: ast.Return) -> None:
        self.return_count += 1
        if node.value is None:
            self._fail("return must include exactly one expression", node)
            return
        if isinstance(node.value, ast.Tuple):
            self._fail("return must be a single tensor expression, not a tuple", node)
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        path = _attribute_path(node.func)
        if path is not None:
            if any(path.startswith(prefix) for prefix in _BANNED_TORCH_PREFIXES):
                self._fail(f"banned torch call: {path}", node)
                return
            if path in _BANNED_TORCH_EXACT:
                self._fail(f"banned torch call: {path}", node)
                return

        if isinstance(node.func, ast.Attribute) and node.func.attr in _BANNED_TENSOR_METHODS:
            self._fail(f"banned tensor method: .{node.func.attr}()", node)
            return

        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _BANNED_CTRL_FLOW):
            self._fail(f"control flow node is not allowed: {type(node).__name__}", node)
            return
        if isinstance(node, _BANNED_EXPR_FORMS):
            self._fail(f"expression form is not allowed: {type(node).__name__}", node)
            return
        super().generic_visit(node)


def _validate_returns(fn: ast.FunctionDef, visitor: _GuardVisitor) -> None:
    if not fn.body or not isinstance(fn.body[-1], ast.Return):
        raise ValidationError("last function statement must be a return")
    if visitor.return_count != 1:
        raise ValidationError("function must contain exactly one return statement")


def validate(pytorch_code: str) -> None:
    try:
        module = ast.parse(pytorch_code)
    except SyntaxError as exc:
        raise ValidationError(f"syntax error line {exc.lineno}: {exc.msg}") from exc

    fn = _validate_module(module)
    _validate_signature(fn)

    visitor = _GuardVisitor()
    for stmt in fn.body:
        visitor.visit(stmt)
        if visitor.error is not None:
            raise ValidationError(visitor.error)

    _validate_returns(fn, visitor)
