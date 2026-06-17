.PHONY: serve test observe observe-docker capture capture-render

# Run the gateway (host, no container).
serve:
	GPU_HOURLY_RATE=$(or $(GPU_HOURLY_RATE),0.80) uvicorn main:app --reload

# Run the test suite (no third-party test deps required).
test:
	python test/test_cost.py
	python test/test_scheduler.py

# Observability on the host — no Docker. Needs: brew install prometheus grafana,
# and the gateway running (make serve) in another shell.
observe:
	./observability/run-local.sh

# Observability via Docker compose (gateway + Prometheus + Grafana containers).
observe-docker:
	cd observability && docker compose up --build

# Drive load and screenshot every Grafana panel to observability/screenshots/.
# Headless-browser capture (Playwright) — works on Apple Silicon, no plugin.
# One time:  pip install playwright && playwright install chromium
capture:
	python observability/capture.py

# Alternative capture via Grafana's render API (x86_64 / Linux only; the
# image-renderer plugin has no darwin-arm64 build).
capture-render:
	./observability/capture.sh
