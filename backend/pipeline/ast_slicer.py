import ast
import os
from typing import Optional

# Tree-sitter is an optional dependency. When missing, non-Python files fall
# back to the context-window slicer. ImportError is caught deliberately here
# because graceful degradation when the package is absent is an explicit
# requirement from issue #3 and approved by the task spec.
try:
    from tree_sitter import Language, Parser as TSParser
    import tree_sitter_javascript
    import tree_sitter_typescript
    import tree_sitter_go
    import tree_sitter_rust

    _JS_LANGUAGE = Language(tree_sitter_javascript.language())
    _TSX_LANGUAGE = Language(tree_sitter_typescript.language_tsx())
    _TS_LANGUAGE = Language(tree_sitter_typescript.language_typescript())
    _GO_LANGUAGE = Language(tree_sitter_go.language())
    _RUST_LANGUAGE = Language(tree_sitter_rust.language())

    _TREE_SITTER_AVAILABLE = True
except (ImportError, TypeError):
    _TREE_SITTER_AVAILABLE = False

# Extensions handled by tree-sitter, mapped to their Language object.
# Defined at module level so they are built once, not per-call.
_EXT_TO_LANGUAGE = {}
if _TREE_SITTER_AVAILABLE:
    _EXT_TO_LANGUAGE = {
        ".js": _JS_LANGUAGE,
        ".jsx": _JS_LANGUAGE,
        ".ts": _TS_LANGUAGE,
        ".tsx": _TSX_LANGUAGE,
        ".go": _GO_LANGUAGE,
        ".rs": _RUST_LANGUAGE,
    }

# Tree-sitter node types that represent a meaningful code block worth
# extracting for each language family.
_JS_TS_BLOCK_TYPES = frozenset(
    {"function_declaration", "method_definition", "arrow_function", "class_declaration"}
)
_GO_BLOCK_TYPES = frozenset(
    {"function_declaration", "method_declaration", "type_declaration"}
)
_RUST_BLOCK_TYPES = frozenset({"function_item", "impl_item", "struct_item"})

_LANGUAGE_BLOCK_TYPES = {}
if _TREE_SITTER_AVAILABLE:
    _LANGUAGE_BLOCK_TYPES = {
        _JS_LANGUAGE: _JS_TS_BLOCK_TYPES,
        _TS_LANGUAGE: _JS_TS_BLOCK_TYPES,
        _TSX_LANGUAGE: _JS_TS_BLOCK_TYPES,
        _GO_LANGUAGE: _GO_BLOCK_TYPES,
        _RUST_LANGUAGE: _RUST_BLOCK_TYPES,
    }


class ASTSlicer:
    """
    Reduces context-window constraints by slicing diff files down to the
    specific AST node (function/class/method) that contains a target line.

    Language routing:
    - .py                 — Python stdlib ast module
    - .js / .jsx          — tree-sitter-javascript
    - .ts / .tsx          — tree-sitter-typescript
    - .go                 — tree-sitter-go
    - .rs                 — tree-sitter-rust
    - everything else     — ±20-line context-window fallback
    - file_path=None      — attempts Python ast, falls back on SyntaxError
    """

    def get_node_at_line(
        self,
        file_content: str,
        line_number: int,
        file_path: Optional[str] = None,
    ) -> str:
        """
        Return the source text of the tightest AST node enclosing line_number.

        Args:
            file_content:  Full text of the file.
            line_number:   1-indexed line number of the changed line.
            file_path:     Optional path (or filename) used to determine the
                           language parser.  When omitted the method tries the
                           Python ast module and falls back to the context
                           window if parsing fails.
        """
        if file_path is None:
            return self._slice_python(file_content, line_number)

        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".py":
            return self._slice_python(file_content, line_number)

        language = _EXT_TO_LANGUAGE.get(extension)
        if language is not None:
            return self._slice_tree_sitter(file_content, line_number, language)

        return self._fallback_slice(file_content, line_number)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _slice_python(self, file_content: str, line_number: int) -> str:
        """Extract the tightest function or class node via Python's ast module."""
        try:
            tree = ast.parse(file_content)
        except SyntaxError:
            return self._fallback_slice(file_content, line_number)

        tightest_node = None
        for node in ast.walk(tree):
            if not (hasattr(node, "lineno") and hasattr(node, "end_lineno")):
                continue
            if not (node.lineno <= line_number <= node.end_lineno):
                continue
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            # Keep the tightest (smallest span) matching node.
            if tightest_node is None or _span(node) < _span(tightest_node):
                tightest_node = node

        if tightest_node is not None:
            lines = file_content.splitlines()
            return "\n".join(lines[tightest_node.lineno - 1 : tightest_node.end_lineno])

        return self._fallback_slice(file_content, line_number)

    def _slice_tree_sitter(
        self, file_content: str, line_number: int, language: "Language"
    ) -> str:
        """
        Extract the tightest matching block node via tree-sitter.

        tree-sitter uses 0-indexed rows; line_number is 1-indexed.
        """
        parser = TSParser(language)
        tree = parser.parse(file_content.encode())
        block_types = _LANGUAGE_BLOCK_TYPES[language]
        # Convert to 0-indexed for tree-sitter comparisons.
        target_row = line_number - 1

        tightest_node = _find_tightest_node(tree.root_node, target_row, block_types)

        if tightest_node is not None:
            lines = file_content.splitlines()
            # tree-sitter end_point row is inclusive.
            return "\n".join(
                lines[tightest_node.start_point[0] : tightest_node.end_point[0] + 1]
            )

        return self._fallback_slice(file_content, line_number)

    def _fallback_slice(
        self, file_content: str, line_number: int, context_lines: int = 20
    ) -> str:
        """Return ±context_lines lines around line_number (1-indexed)."""
        lines = file_content.splitlines()
        start = max(0, line_number - 1 - context_lines)
        end = min(len(lines), line_number + context_lines)
        return "\n".join(lines[start:end])


# ------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ------------------------------------------------------------------

def _span(node: ast.AST) -> int:
    """Line span of a Python AST node — smaller means tighter."""
    return node.end_lineno - node.lineno  # type: ignore[attr-defined]


def _find_tightest_node(root, target_row: int, block_types: frozenset):
    """
    Walk a tree-sitter tree and return the tightest node whose type is in
    block_types and that contains target_row (0-indexed).

    Returns None if no matching node is found.
    """
    tightest = None
    stack = [root]
    while stack:
        node = stack.pop()
        start_row = node.start_point[0]
        end_row = node.end_point[0]
        if not (start_row <= target_row <= end_row):
            continue
        if node.type in block_types:
            if tightest is None or (end_row - start_row) < (
                tightest.end_point[0] - tightest.start_point[0]
            ):
                tightest = node
        stack.extend(node.children)
    return tightest
