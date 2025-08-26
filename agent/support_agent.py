import json
import os
from typing import Optional

# Try to support both the modern and legacy OpenAI SDKs
try:
    from openai import OpenAI
    _OPENAI_NEW_SDK = True
except Exception:
    import openai
    _OPENAI_NEW_SDK = False

class SupportAgent:
    def __init__(self, kb_path: str = "agent/knowledge_base.json"):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)

    def _kb_lookup(self, query: str) -> Optional[str]:
        ql = query.strip().lower()
        if not ql:
            return None
        # exact or substring match over KB keys
        for k, v in self.kb.items():
            kl = k.lower()
            if kl == ql or kl in ql or ql in kl:
                return v
        return None

    def _call_llm(self, query: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if not api_key:
            return "(No OPENAI_API_KEY set) I cannot call the LLM. Please set an environment variable or use the KB."

        try:
            if _OPENAI_NEW_SDK:
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system", "content":"You are a concise, helpful healthcare support assistant."},
                        {"role":"user", "content": query}
                    ],
                    temperature=0.2,
                )
                return resp.choices[0].message.content
            else:
                openai.api_key = api_key
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system", "content":"You are a concise, helpful healthcare support assistant."},
                        {"role":"user", "content": query}
                    ],
                    temperature=0.2,
                )
                return resp["choices"][0]["message"]["content"]
        except Exception as e:
            return f"(LLM error) {e}"

    def get_answer(self, query: str) -> str:
        # 1) Try knowledge base
        kb_ans = self._kb_lookup(query)
        if kb_ans:
            return kb_ans

        # 2) Fallback to LLM (if key exists)
        llm_ans = self._call_llm(query)
        # Basic fallback/escalation decision
        if llm_ans.startswith("(No OPENAI_API_KEY") or llm_ans.startswith("(LLM error"):
            return "I couldn't find a definitive answer — I'll escalate you to a human support agent."
        return llm_ans
