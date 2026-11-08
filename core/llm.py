import os
import json
import requests

class OllamaClient:
    def __init__(self):
        self.base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.enabled = os.getenv("LLM_ENABLED", "true").lower() == "true"

    def chat(self, prompt):
        if not self.enabled:
            return None
        try:
            r = requests.post(
                f"{self.base}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=90,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception:
            return None

    def plan(self, question, schema, history):
        prompt = f"""You are the planning component of a data analyst.
Return ONLY valid JSON with keys:
intent, dataset, group_by, metric, aggregation, time_column, top_n, chart.
Allowed aggregation: sum, mean, count, min, max.
Allowed chart: bar, line, pie, scatter, none.
Question: {question}
Schema:
{schema}
Recent conversation:
{history}
If a field is unknown, use null. Do not invent column names."""
        raw = self.chat(prompt)
        if not raw:
            return None
        try:
            raw = raw.strip().replace("```json", "").replace("```", "")
            return json.loads(raw)
        except Exception:
            return None
