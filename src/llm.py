"""Cliente único para qualquer LLM com API compatível com a da OpenAI.

Groq, Gemini e Ollama expõem esse formato, então trocar de provedor é só
mudar variáveis de ambiente — sem reescrever código.

  LLM_PROVIDER = groq | gemini | ollama | custom   (padrão: groq)
  LLM_MODEL    = sobrescreve o modelo padrão do provedor
  LLM_API_KEY  = chave (ou use GROQ_API_KEY / GEMINI_API_KEY)
  LLM_BASE_URL = obrigatório só para "custom"

Os nomes de modelos mudam com frequência: se um padrão abaixo deixar de existir,
defina LLM_MODEL com um modelo listado no painel do provedor.
"""
import os
import time

from openai import OpenAI

PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "model": "llama-3.3-70b-versatile"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY",
               "model": "gemini-2.5-flash"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None, "model": "qwen2.5:3b"},
    "custom": {"base_url": None, "key_env": None, "model": None},
}


class LLM:
    def __init__(self, provider=None, model=None, temperature=0.0):
        self.provider = provider or os.environ.get("LLM_PROVIDER", "groq")
        cfg = PROVIDERS[self.provider]
        base_url = os.environ.get("LLM_BASE_URL") or cfg["base_url"]
        key = os.environ.get("LLM_API_KEY") or (os.environ.get(cfg["key_env"]) if cfg["key_env"] else None)
        if not base_url:
            raise SystemExit("Defina LLM_BASE_URL para o provedor 'custom'.")
        if cfg["key_env"] and not key:
            raise SystemExit(f"Defina {cfg['key_env']} (ou LLM_API_KEY) para usar o provedor '{self.provider}'.")
        self.model = model or os.environ.get("LLM_MODEL") or cfg["model"]
        if not self.model:
            raise SystemExit("Defina LLM_MODEL.")
        self.client = OpenAI(base_url=base_url, api_key=key or "not-needed")
        self.temperature = temperature

    def chat(self, system: str, user: str, max_tokens=700, retries=4) -> str:
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # limites de taxa dos planos gratuitos: espera e tenta de novo
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt * 5
                print(f"  [llm] {type(e).__name__}: {e}. Nova tentativa em {wait}s…")
                time.sleep(wait)

    def __repr__(self):
        return f"{self.provider}:{self.model}"
