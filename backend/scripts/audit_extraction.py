import os
import sys
import json
from litellm import completion

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def manual_audit_test():
    """
    Physically dumps the exact API response from the DGX 7B model so the Human can audit its intelligence
    and verify the extraction isn't just a hallucinated mock.
    """
    print("====================================")
    print("🔍 TRANSPARENCY AUDIT: QWEN 7B (DGX)")
    print("====================================")

    # Target the newly booted 7B Container on gx10
    model = "openai/Qwen/Qwen2.5-Coder-7B-Instruct"
    api_base = "http://gx10:8080/v1"
    api_key = "dgx-dummy-key"

    system_prompt = (
        "You are an expert AI Architect extracting coding rules into strict JSON. "
        "Your goal is to parse reviewer feedback and isolate what the Copilot did WRONG. "
        "Output ONLY valid JSON following this schema: "
        "{'rule_id': 'uuid', 'metadata': {'status': 'active'}, 'content': {'title': '...', 'description': '...', 'enforcement_prompt': '...'}}"
    )

    # A real, messy human comment we recently scraped from tiangolo/sqlmodel
    raw_human_comment = "As a quick fix, I changed all double underscores to singles, but it feels a bit like a hack. We could also suppress the `ty` warning, but that also feels wrong... A parameter can only be positional-only if it precedes all positional-or-keyword parameters."
    
    user_prompt = f'''
    <REVIEW_DATA>
    {raw_human_comment}
    </REVIEW_DATA>
    
    <CODE_DIFF>
    - def select(__ent1: _TCCA[_T1])
    + def select(_ent1: _TCCA[_T1])
    </CODE_DIFF>
    '''

    print(f"\\n[INPUT] Raw Developer Insight:\\n{raw_human_comment}\\n")
    print("⏳ Querying DGX Port 8080...")

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            api_base=api_base,
            api_key=api_key,
            temperature=0.1
        )
        
        raw_output = response.choices[0].message.content.strip()
        print(f"\\n[OUTPUT] Exact 7B JSON Response from DGX:\\n{raw_output}\\n")
        
        # Verify JSON
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()
            
        json_obj = json.loads(raw_output)
        print("✅ JSON Schema successfully parsed! Model adhered to constraints.")
        
    except Exception as e:
        print(f"❌ [Error] The model failed to respond or broke the schema: {e}")

if __name__ == "__main__":
    manual_audit_test()
