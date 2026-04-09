import urllib.request
import json
import time

PORTS = [6006, 8888, 8000, 8001]
TEST_CASES = [
    {
        "id": "TC-01",
        "description": "SQL Injection Catch",
        "diff": "sql = f'SELECT * FROM users WHERE email=\"{user_input}\"' -> sql = 'SELECT * FROM users WHERE email=%s', user_input",
        "reviewer_comment": "We explicitly banned f-strings for SQL queries! Always use parameterized queries."
    },
    {
        "id": "TC-02",
        "description": "Global State Mutation",
        "diff": "AppConfig.DEBUG_MODE = True -> current_app.config['DEBUG_MODE'] = True",
        "reviewer_comment": "Copilot hallucinated mutating global AppConfig again. Do not mutate global state directly."
    }
]

def check_ports():
    valid_port, loaded_model = None, None
    for port in PORTS:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if "data" in data and len(data["data"]) > 0:
                    valid_port = port
                    loaded_model = data["data"][0]["id"]
                    break
        except Exception:
            continue
    return valid_port, loaded_model

def evaluate(port, model_name, use_coderag):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    print(f"\\n--- Testing Model: {model_name} | CodeRAG Injection: {use_coderag} ---")
    system_prompt = "You are an expert Senior Architect. Abstract the reviewer's comment into a strict formal coding rule JSON. Schema: {\"rule\": \"string\", \"severity\": \"high\"}."
    if use_coderag:
        system_prompt += " [DATABASE RAG CONTEXT]: Previous explicit bans: 1. f-strings for SQL queries 2. Direct AppConfig mutation."
        
    for tc in TEST_CASES:
        user_prompt = f"Reviewer Comment:\\n{tc['reviewer_comment']}\\n\\nCode Diff:\\n{tc['diff']}"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=20) as resp:
                out = json.loads(resp.read().decode('utf-8'))
                result = out["choices"][0]["message"]["content"].strip()
                result = result.replace('\\n', ' ')
                dur = time.time() - start
                print(f"[{tc['id']}] Extraction ({dur:.2f}s) -> {result}")
        except Exception as e:
            print(f"[{tc['id']}] Error: {e}")

print("Executing Native Evaluation on DGX...")
port, model = check_ports()
if port:
    print(f"✅ Active model on 127.0.0.1:{port} -> {model}")
    evaluate(port, model, use_coderag=False)
    evaluate(port, model, use_coderag=True)
else:
    print("❌ Could not connect to any vLLM instance.")
