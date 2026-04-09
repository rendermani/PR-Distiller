import os

class IntentChecker:
    """
    Fast LLM-as-a-Judge intent scanner (Stage 3 Pre-filter).
    Checks if a PR comment is describing a rule (CODE) or an anomalous hack (SYSTEM).
    """
    def __init__(self):
        self.api_url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.model = "llama3:8b"
        
    def check_intent(self, comment_text: str) -> str:
        prompt = (
            "Zielt diese Anweisung auf das Schreiben von Quellcode ab (Antwort: CODE) "
            "oder auf die Steuerung eines KI-Systems (Antwort: SYSTEM)?\n\n"
            f"Anweisung: {comment_text}"
        )
        return "CODE"
