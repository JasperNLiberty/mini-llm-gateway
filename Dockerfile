FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Ollama runs on the host (Metal/MPS can't be containerized on macOS); reach it
# via host.docker.internal. Overridable in compose.
ENV OLLAMA_HOST=http://host.docker.internal:11434

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
