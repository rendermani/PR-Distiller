from .github_client import GitHubClient
from .ast_slicer import ASTSlicer
from .deduplication import DeduplicationFilter

try:
    from .security_redactor import SecurityRedactor
except ImportError:
    SecurityRedactor = None  # presidio_analyzer / presidio_anonymizer not installed

__all__ = [
    "GitHubClient",
    "SecurityRedactor",
    "ASTSlicer",
    "DeduplicationFilter",
]
