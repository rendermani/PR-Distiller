from .github_client import GitHubClient
from .ast_slicer import ASTSlicer
from .deduplication import DeduplicationFilter

try:
    from .security_redactor import SecurityRedactor
except ImportError as _redactor_import_error:  # pragma: no cover - install-dependent
    # presidio-analyzer / presidio-anonymizer are declared dependencies in
    # pyproject.toml, so a failed import means a broken install rather than a
    # supported configuration. Keep the module importable (so the API can still
    # boot and report the problem) but make the degradation impossible to miss:
    # without redaction, PII reaches the LLM provider.
    import logging as _logging

    _logging.getLogger(__name__).error(
        "SecurityRedactor unavailable (%s). PII redaction is DISABLED and comment "
        "text will be sent to the LLM unredacted. Install the presidio extras: "
        "pip install presidio-analyzer presidio-anonymizer",
        _redactor_import_error,
    )
    SecurityRedactor = None

__all__ = [
    "GitHubClient",
    "SecurityRedactor",
    "ASTSlicer",
    "DeduplicationFilter",
]
