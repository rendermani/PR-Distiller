import ast

class ASTSlicer:
    """
    Reduces Context Window constraints by programmatically slicing
    diff files down to the specific AST node (Function/Class) modified.
    """
    
    def get_node_at_line(self, file_content: str, line_number: int) -> str:
        """
        Extracts only the code chunk corresponding to the given line number.
        """
        try:
            tree = ast.parse(file_content)
        except SyntaxError:
            # Fallback for non-Python or broken syntax files
            # For a fully robust system, tree-sitter would be used for multi-language support.
            return self._fallback_slice(file_content, line_number)
            
        target_node = None
        # Walk through the AST to find the tightest wrapping function or class
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                if node.lineno <= line_number <= node.end_lineno:
                    # We capture Functions or Classes to provide enough context
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        target_node = node
                        
        if target_node:
            lines = file_content.splitlines()
            return "\n".join(lines[target_node.lineno-1:target_node.end_lineno])
            
        # If no specific block found, return fallback
        return self._fallback_slice(file_content, line_number)

    def _fallback_slice(self, file_content: str, line_number: int, context_lines: int = 20) -> str:
        """Fallback that just returns N lines above and below the target line."""
        lines = file_content.splitlines()
        start = max(0, line_number - 1 - context_lines)
        end = min(len(lines), line_number + context_lines)
        return "\n".join(lines[start:end])
