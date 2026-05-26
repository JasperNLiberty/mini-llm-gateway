import httpx

OLLAMA_URL = "http://localhost:11434"

class OllamaClient:
    def __init__(self, base_url=OLLAMA_URL):
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(self, model: str, prompt: str) -> str:
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]

    async def close(self):
        await self.client.aclose()
