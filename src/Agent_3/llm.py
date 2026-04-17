"""
llm.py — Ollama client for Agent_3.
"""

from __future__ import annotations

import os
import re
import time

import requests


OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "qwen2.5-coder:14b"
TIMEOUT = int(os.environ.get("DISASTER_AGENT_LLM_TIMEOUT", "1000"))


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._check_connection()

    def _check_connection(self) -> None:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            if not models:
                print("[LLM] WARNING: Ollama is running but no models are pulled.")
            else:
                print(f"[LLM] Connected to Ollama. Available models: {models}")
                if self.model not in models and not any(self.model in m for m in models):
                    print(f"[LLM] WARNING: model '{self.model}' not found. Available: {models}")
        except requests.exceptions.ConnectionError:
            print("[LLM] ERROR: Cannot connect to Ollama at localhost:11434")
            raise

    def _call(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "stream": False,
        }
        try:
            preview = user.strip().splitlines()[0][:100] if user.strip() else "(empty prompt)"
            started = time.perf_counter()
            print(f"[LLM] Request started | model={self.model} | timeout={TIMEOUT}s | prompt='{preview}'")
            response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            elapsed = time.perf_counter() - started
            print(f"[LLM] Request completed in {elapsed:.1f}s")
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            return f"[LLM ERROR] Request timed out after {TIMEOUT} seconds"
        except Exception as exc:  # noqa: BLE001
            return f"[LLM ERROR] {exc}"

    def propose(self, system: str, user: str) -> tuple[str, str]:
        response = self._call(system, user)
        if response.startswith("[LLM ERROR]"):
            print(f"[LLM] Error: {response}")
            return "", ""
        return response, extract_code_block(response)

    def respond(self, system: str, user: str) -> str:
        response = self._call(system, user)
        if response.startswith("[LLM ERROR]"):
            print(f"[LLM] Error: {response}")
            return ""
        return response

    def analyze(self, prompt: str) -> str:
        response = self._call("You are a concise ML research analyst.", prompt)
        if response.startswith("[LLM ERROR]"):
            return "Analysis unavailable (LLM error)."
        return response


def extract_code_block(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""
