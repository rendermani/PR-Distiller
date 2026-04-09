import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class SecurityRedactor:
    """
    Programmatic filter layer executed BEFORE data hits any LLM.
    Ensures secrets and PII are redacted from context.
    """
    def __init__(self):
        # Initialize Microsoft Presidio engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Example programmatic Regex for Hardcoded Secrets
        # Matches typical 20-character AWS Access Key IDs
        self.aws_key_pattern = re.compile(r'(?<![A-Z0-9])[A-Z0-9]{20}(?![A-Z0-9])')

    def redact_text(self, text: str) -> str:
        if not text:
            return ""
            
        # 1. Hardcoded Secrets Redaction (Regex / Gitleaks logic)
        redacted = self.aws_key_pattern.sub('[REDACTED_AWS_KEY]', text)
        
        # 2. PII / IP / Email Redaction (Presidio NLP)
        results = self.analyzer.analyze(
            text=redacted, 
            entities=["EMAIL_ADDRESS", "IP_ADDRESS"], 
            language='en'
        )
        anonymized_result = self.anonymizer.anonymize(
            text=redacted, 
            analyzer_results=results
        )
        
        return anonymized_result.text
