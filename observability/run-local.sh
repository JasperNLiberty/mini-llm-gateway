#!/usr/bin/env bash
#
# Run Prometheus + Grafana on the host — no Docker — pointed at the gateway on
# localhost:8000. Reuses the exact same dashboard JSON as the Docker stack; only
# the datasource URL (localhost:9090) and scrape target (localhost:8000) differ,
# so there's no host.docker.internal networking to worry about.
#
# Prereqs:
#   brew install prometheus grafana
#   # and have the gateway + Ollama running on the host:
#   ollama serve
#   GPU_HOURLY_RATE=0.80 uvicorn main:app   # (from the repo root)
#
# Then:
#   ./observability/run-local.sh
#   # Grafana :3000 (anon admin)   Prometheus :9090   Gateway :8000
#   # Ctrl-C stops both.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL="$HERE/.local"

# --- preflight -------------------------------------------------------------
command -v prometheus >/dev/null 2>&1 || { echo "Missing prometheus. Run: brew install prometheus"; exit 1; }

# Grafana ships as `grafana server` (modern) or `grafana-server` (older).
if command -v grafana >/dev/null 2>&1; then
  GRAFANA=(grafana server)
elif command -v grafana-server >/dev/null 2>&1; then
  GRAFANA=(grafana-server)
else
  echo "Missing grafana. Run: brew install grafana"; exit 1
fi
GRAFANA_HOME="$(brew --prefix grafana 2>/dev/null)/share/grafana"

# --- generate host-local provisioning (localhost URL + absolute dash path) --
PROV="$LOCAL/provisioning"
mkdir -p "$PROV/datasources" "$PROV/dashboards" \
         "$LOCAL/prometheus-data" "$LOCAL/grafana-data" "$LOCAL/grafana-logs"

cat > "$PROV/datasources/datasource.yml" <<YAML
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    uid: prometheus
    isDefault: true
    editable: false
YAML

cat > "$PROV/dashboards/dashboards.yml" <<YAML
apiVersion: 1
providers:
  - name: mini-llm-gateway
    type: file
    disableDeletion: false
    editable: true
    options:
      path: $HERE/grafana/dashboards
YAML

# --- start prometheus ------------------------------------------------------
prometheus \
  --config.file="$HERE/prometheus.local.yml" \
  --storage.tsdb.path="$LOCAL/prometheus-data" \
  --web.listen-address="127.0.0.1:9090" &
PROM_PID=$!

# --- start grafana (all writable paths under .local; provisioning from PROV)-
GF_PATHS_DATA="$LOCAL/grafana-data" \
GF_PATHS_LOGS="$LOCAL/grafana-logs" \
GF_PATHS_PROVISIONING="$PROV" \
GF_AUTH_ANONYMOUS_ENABLED=true \
GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
GF_AUTH_DISABLE_LOGIN_FORM=true \
GF_SERVER_HTTP_PORT=3000 \
  "${GRAFANA[@]}" --homepath "$GRAFANA_HOME" &
GF_PID=$!

cleanup() { kill "$PROM_PID" "$GF_PID" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

echo ""
echo "  Prometheus  http://localhost:9090"
echo "  Grafana     http://localhost:3000   (anonymous admin, no login)"
echo "  Gateway     http://localhost:8000   (start it separately if not running)"
echo "  Ctrl-C to stop both."
echo ""
wait
