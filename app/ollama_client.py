import json
import os

import httpx

# Configurable so a containerized gateway can reach Ollama on the host
# (OLLAMA_HOST=http://host.docker.internal:11434). Default keeps local runs
# unchanged.
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

class OllamaClient:
    def __init__(self, base_url=OLLAMA_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(self, model: str, prompt: str, max_tokens: int = 256) -> dict:
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            }
        )
        data = response.json()
        if "error" in data:
            raise Exception(data["error"])

        # Ollama reports timing in nanoseconds. eval_duration covers only the
        # generation phase, so it yields a clean output tokens/sec figure that
        # excludes prompt processing and queueing.
        output_tokens = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        tokens_per_sec = (
            output_tokens / (eval_duration_ns / 1e9)
            if eval_duration_ns > 0
            else 0.0
        )

        return {
            "response": data["response"],
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": output_tokens,
            "tokens_per_sec": tokens_per_sec,
            # kept for backward compatibility with existing metrics
            "tokens": output_tokens,
        }

    async def chat(self, model: str, prompt: str, max_tokens: int = 1024,
                   think: bool = True) -> dict:
        """Non-streaming /api/chat with optional thinking.

        For a reasoning model with ``think=True``, Ollama returns the hidden
        reasoning in ``message.thinking`` separately from ``message.content``,
        which is what lets the gateway report the thinking/answer split. Ollama
        only gives a combined ``eval_count``, so per-part token counts are
        apportioned by character length (a good proxy at the chat level).
        """
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if think:
            payload["think"] = True
        response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
        data = response.json()
        if "error" in data:
            raise Exception(data["error"])

        msg = data.get("message", {}) or {}
        thinking = msg.get("thinking") or ""
        answer = msg.get("content") or ""

        output_tokens = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        tokens_per_sec = (
            output_tokens / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
        )

        # Apportion total output tokens into thinking vs answer by char length.
        think_chars, ans_chars = len(thinking), len(answer)
        total_chars = think_chars + ans_chars
        thinking_tokens = (
            round(output_tokens * think_chars / total_chars) if total_chars else 0
        )
        answer_tokens = output_tokens - thinking_tokens

        return {
            "response": answer,
            "thinking": thinking,
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "answer_tokens": answer_tokens,
            "tokens_per_sec": tokens_per_sec,
            "tokens": output_tokens,  # compat with existing token metric
        }

    async def generate_stream(self, model: str, prompt: str, max_tokens: int = 256):
        """Stream raw Ollama NDJSON chunks as they arrive.

        Yields each decoded chunk dict. Token chunks carry a ``response`` delta;
        the final chunk has ``done: true`` plus the authoritative timing/usage
        fields (``eval_count``, ``eval_duration``, ``prompt_eval_count``). The
        caller is responsible for accumulating text and reading the final stats.
        """
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": max_tokens},
            },
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise Exception(data["error"])
                yield data

    async def close(self):
        await self.client.aclose()
