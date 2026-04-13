"""
Tests for issue #3: Multi-language AST support via tree-sitter.

Covers:
- Python files still use ast.parse (existing behavior preserved)
- JS function extraction at a given line
- TS class/method extraction
- Go function extraction
- Rust function/impl extraction
- Unknown extension falls back to context window
- file_path=None preserves backward compatibility (tries Python ast first)
- Graceful degradation if tree-sitter is not installed (mock the import)
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load ast_slicer directly to avoid pipeline/__init__.py pulling in optional deps
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "pipeline.ast_slicer",
    os.path.join(os.path.dirname(__file__), "..", "pipeline", "ast_slicer.py"),
)
_ast_slicer_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ast_slicer_mod)
ASTSlicer = _ast_slicer_mod.ASTSlicer


# ---------------------------------------------------------------------------
# Fixtures — minimal but valid source files for each language
# ---------------------------------------------------------------------------

PYTHON_SOURCE = """\
def unrelated():
    pass

def target_function(x, y):
    result = x + y
    return result

class SomeClass:
    pass
"""

JS_SOURCE = """\
function unrelated() {
  return 0;
}

function targetFunction(x, y) {
  const result = x + y;
  return result;
}
"""

TS_SOURCE = """\
function helper(): void {
  console.log("help");
}

class TargetClass {
  method(x: number): number {
    return x * 2;
  }
}
"""

GO_SOURCE = """\
package main

func unrelated() int {
\treturn 0
}

func targetFunction(x int, y int) int {
\treturn x + y
}
"""

RUST_SOURCE = """\
fn unrelated() -> i32 {
    0
}

fn target_function(x: i32, y: i32) -> i32 {
    x + y
}
"""

UNKNOWN_SOURCE = """\
line 1
line 2
line 3
line 4
line 5
"""


class TestPythonBehaviorPreserved(unittest.TestCase):
    """Python files must still use ast.parse — no regression."""

    def setUp(self):
        self.slicer = ASTSlicer()

    def test_python_file_extracts_function_containing_line(self):
        # Line 5 is inside target_function (lines 4-6)
        result = self.slicer.get_node_at_line(PYTHON_SOURCE, 5, file_path="module.py")
        self.assertIn("target_function", result)
        self.assertNotIn("unrelated", result)

    def test_python_file_extracts_class_containing_line(self):
        # Line 9 is inside SomeClass
        result = self.slicer.get_node_at_line(PYTHON_SOURCE, 9, file_path="path/to/module.py")
        self.assertIn("SomeClass", result)

    def test_backward_compat_no_file_path_uses_python_ast(self):
        """Calling without file_path must behave identically to before."""
        result_with = self.slicer.get_node_at_line(PYTHON_SOURCE, 5, file_path="m.py")
        result_without = self.slicer.get_node_at_line(PYTHON_SOURCE, 5)
        self.assertEqual(result_with, result_without)

    def test_no_file_path_falls_back_for_non_python_syntax(self):
        """file_path=None on JS source: ast.parse fails, use fallback."""
        result = self.slicer.get_node_at_line(JS_SOURCE, 5)
        # Fallback returns context window — must include some content
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestJavaScriptExtraction(unittest.TestCase):
    """JS files (.js, .jsx) must use tree-sitter."""

    def setUp(self):
        self.slicer = ASTSlicer()

    def test_js_extracts_function_at_line(self):
        # Line 5 is the start of targetFunction
        result = self.slicer.get_node_at_line(JS_SOURCE, 5, file_path="app.js")
        self.assertIn("targetFunction", result)

    def test_js_does_not_include_unrelated_function(self):
        result = self.slicer.get_node_at_line(JS_SOURCE, 6, file_path="component.jsx")
        self.assertNotIn("unrelated", result)

    def test_jsx_extension_uses_js_parser(self):
        result = self.slicer.get_node_at_line(JS_SOURCE, 5, file_path="component.jsx")
        self.assertIn("targetFunction", result)


class TestTypeScriptExtraction(unittest.TestCase):
    """TS files (.ts, .tsx) must use tree-sitter."""

    def setUp(self):
        self.slicer = ASTSlicer()

    def test_ts_extracts_class_at_line(self):
        # Line 5 is the start of TargetClass
        result = self.slicer.get_node_at_line(TS_SOURCE, 5, file_path="service.ts")
        self.assertIn("TargetClass", result)

    def test_ts_method_line_returns_enclosing_class_or_method(self):
        # Line 6 is inside method() — should return the method or class
        result = self.slicer.get_node_at_line(TS_SOURCE, 6, file_path="service.ts")
        self.assertIn("method", result)

    def test_tsx_extension_uses_ts_parser(self):
        result = self.slicer.get_node_at_line(TS_SOURCE, 5, file_path="Component.tsx")
        self.assertIn("TargetClass", result)


class TestGoExtraction(unittest.TestCase):
    """Go files (.go) must use tree-sitter."""

    def setUp(self):
        self.slicer = ASTSlicer()

    def test_go_extracts_function_at_line(self):
        # Line 7 is the start of targetFunction
        result = self.slicer.get_node_at_line(GO_SOURCE, 7, file_path="main.go")
        self.assertIn("targetFunction", result)

    def test_go_does_not_include_unrelated_function(self):
        result = self.slicer.get_node_at_line(GO_SOURCE, 8, file_path="main.go")
        self.assertNotIn("unrelated", result)


class TestRustExtraction(unittest.TestCase):
    """Rust files (.rs) must use tree-sitter."""

    def setUp(self):
        self.slicer = ASTSlicer()

    def test_rust_extracts_function_at_line(self):
        # Line 5 is the start of target_function
        result = self.slicer.get_node_at_line(RUST_SOURCE, 5, file_path="lib.rs")
        self.assertIn("target_function", result)

    def test_rust_does_not_include_unrelated_function(self):
        result = self.slicer.get_node_at_line(RUST_SOURCE, 6, file_path="lib.rs")
        self.assertNotIn("fn unrelated", result)


class TestUnknownExtensionFallback(unittest.TestCase):
    """Unknown extensions must use the context-window fallback."""

    def setUp(self):
        self.slicer = ASTSlicer()

    def test_unknown_extension_returns_context_window(self):
        result = self.slicer.get_node_at_line(UNKNOWN_SOURCE, 3, file_path="data.yaml")
        # Must contain the target line
        self.assertIn("line 3", result)

    def test_no_extension_returns_context_window(self):
        result = self.slicer.get_node_at_line(UNKNOWN_SOURCE, 3, file_path="Makefile")
        self.assertIn("line 3", result)


class TestGracefulDegradationWithoutTreeSitter(unittest.TestCase):
    """When tree-sitter is unavailable, non-Python files must fall back gracefully."""

    def test_js_falls_back_when_tree_sitter_unavailable(self):
        """When _TREE_SITTER_AVAILABLE is False, JS files must use _fallback_slice."""
        # Temporarily flip the module-level flag off
        orig = _ast_slicer_mod._TREE_SITTER_AVAILABLE
        orig_ext = _ast_slicer_mod._EXT_TO_LANGUAGE.copy()
        try:
            _ast_slicer_mod._TREE_SITTER_AVAILABLE = False
            _ast_slicer_mod._EXT_TO_LANGUAGE = {}
            slicer = ASTSlicer()
            result = slicer.get_node_at_line(JS_SOURCE, 5, file_path="app.js")
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
        finally:
            _ast_slicer_mod._TREE_SITTER_AVAILABLE = orig
            _ast_slicer_mod._EXT_TO_LANGUAGE = orig_ext


if __name__ == "__main__":
    unittest.main()
