#!/usr/bin/env bash
#
# One-command dashboard screenshots: drive load, then pull each Grafana panel as
# a PNG via the render API. Assumes the stack is already up (make serve + make
# observe) with Ollama running on the host.
#
#   ./observability/capture.sh          # 60s load, then capture to screenshots/
#   DURATION=120 MODEL=llama3.2:1b ./observability/capture.sh
#
# NOTE: the render API needs Grafana's image-renderer plugin (a one-time
# install, printed below if missing). This script never installs Chromium for
# you — it tells you the single command to run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAFANA="${GRAFANA_URL:-http://localhost:3000}"
GATEWAY="${GATEWAY_URL:-http://localhost:8000}"
DASH_UID="llm-gateway"
OUT="$HERE/screenshots"
MODEL="${MODEL:-qwen2.5:7b}"
DURATION="${DURATION:-60}"

mkdir -p "$OUT"

# --- preflight -------------------------------------------------------------
curl -sf "$GATEWAY/healthz" >/dev/null 2>&1 \
  || { echo "Gateway not up at $GATEWAY — run 'make serve' (and 'ollama serve')."; exit 1; }
curl -sf "$GRAFANA/api/health" >/dev/null 2>&1 \
  || { echo "Grafana not up at $GRAFANA — run 'make observe'."; exit 1; }

if ! curl -sf "$GRAFANA/api/plugins/grafana-image-renderer/settings" >/dev/null 2>&1; then
  cat <<MSG
Grafana image-renderer plugin not found — the render API needs it.
Install it into the same local plugins dir 'make observe' uses, then restart it:

  mkdir -p "$HERE/.local/grafana-plugins"
  grafana cli --homepath "\$(brew --prefix grafana)/share/grafana" --pluginsDir "$HERE/.local/grafana-plugins" plugins install grafana-image-renderer
  # stop 'make observe' (Ctrl-C), then run 'make observe' again

Then re-run this script. (Alternatively, capture the panels manually from
$GRAFANA — the dashboard is already provisioned.)
MSG
  exit 2
fi

# --- drive load in the background ------------------------------------------
python3 "$HERE/loadgen.py" --gateway "$GATEWAY" --model "$MODEL" --duration "$DURATION" &
LOAD_PID=$!
trap 'kill "$LOAD_PID" 2>/dev/null || true' EXIT

# Capture near the end of the run, while traffic is still flowing, so the
# time-series panels are populated. Give Prometheus a few scrapes of lead-in.
lead=$(( DURATION - 10 ))
[ "$lead" -lt 20 ] && lead=20
echo "warming up ${lead}s before capture..."
sleep "$lead"

# --- pull each panel + the full board --------------------------------------
FROM="now-15m"; TO="now"; W=1100; H=460
echo "capturing panels -> $OUT"
for p in 1 2 3 4 5 6 7 8 9 10; do
  if curl -sf "$GRAFANA/render/d-solo/$DASH_UID/llm-gateway?orgId=1&panelId=$p&from=$FROM&to=$TO&width=$W&height=$H&theme=light&tz=UTC" -o "$OUT/panel-$p.png"; then
    echo "  saved panel-$p.png"
  else
    echo "  FAILED panel-$p"
  fi
done
curl -sf "$GRAFANA/render/d/$DASH_UID/llm-gateway?orgId=1&from=$FROM&to=$TO&width=1400&height=1300&theme=light&tz=UTC" \
  -o "$OUT/dashboard-full.png" && echo "  saved dashboard-full.png" || echo "  FAILED full board"

wait "$LOAD_PID" 2>/dev/null || true
echo "done — screenshots in $OUT"
