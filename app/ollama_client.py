import httpx

OLLAMA_URL = "http://localhost:11434"

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

    async def close(self):
        await self.client.aclose()
