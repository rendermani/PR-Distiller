import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Initializing Native Transformers Pipeline for Physical POC...")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device mapped to: {device}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

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

def evaluate(use_coderag):
    print(f"\\n--- Testing Model: {model_name} | CodeRAG Database Injection: {use_coderag} ---")
    system_prompt = "You are an expert Senior Architect. Abstract the reviewer's comment into a strict formal coding rule JSON. Schema: {\"rule\": \"string\", \"severity\": \"high\"}."
    if use_coderag:
        system_prompt += " [DATABASE RAG CONTEXT]: Previous explicit bans: 1. f-strings for SQL queries 2. Direct AppConfig mutation."
        
    for tc in TEST_CASES:
        user_prompt = f"Reviewer Comment:\\n{tc['reviewer_comment']}\\n\\nCode Diff:\\n{tc['diff']}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        start = time.time()
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=100, temperature=0.1)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        dur = time.time() - start
        
        # Clean formatting
        out = out.replace('\\n', ' ').strip()
        print(f"[{tc['id']}] Extraction ({dur:.2f}s) -> {out}")

evaluate(use_coderag=False)
evaluate(use_coderag=True)
print("\\n✅ POC Completed Successfully.")
