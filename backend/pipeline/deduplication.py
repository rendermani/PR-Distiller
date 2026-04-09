import hashlib

class DeduplicationFilter:
    """
    Extremely fast clustering and deduplication of PR comments.
    Prevents the LLM / Vector DB from processing identical Copilot/Linter suggestions.
    """
    def __init__(self):
        self.seen_hashes = set()
        
    def _compute_hash(self, text: str) -> str:
        """
        A surrogate for MinHash/LSH that hashes normalized text.
        Strips whitespace and converts to lowercase to group similar texts.
        """
        normalized = "".join(text.split()).lower()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
        
    def is_duplicate(self, text: str) -> bool:
        """Returns True if this exact comment has been seen."""
        text_hash = self._compute_hash(text)
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False
        
    def reset(self):
        """Reset the cache (e.g., between repository scans)."""
        self.seen_hashes.clear()
