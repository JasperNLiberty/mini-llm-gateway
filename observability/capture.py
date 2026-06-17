#!/usr/bin/env python3
"""Headless-browser dashboard capture via Playwright.

Works on Apple Silicon (and everywhere else) with no Grafana image-renderer
plugin and no Docker — it loads the already-provisioned dashboard in a real
headless Chromium and screenshots it. This is the arm64-friendly replacement for
the render-API path in capture.sh (that plugin has no darwin-arm64 build).

Setup (one time):
    pip install playwright
    playwright install chromium

Usage (with the stack up: make serve + make observe):
    python observability/capture.py                      # 60s load, then capture
    python observability/capture.py --no-load            # capture current state
    python observability/capture.py --duration 120 --model llama3.2:1b
    python observability/capture.py --theme dark
"""

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "screenshots"

# Dashboard uid is "llm-gateway" (see grafana/dashboards/llm-gateway.json).
DASH_FULL = "/d/llm-gateway/llm-gateway"
DASH_SOLO = "/d-solo/llm-gateway/llm-gateway"
PANELS = list(range(1, 11))  # panel ids 1..10


def _wait_healthy(grafana: str, gateway: str, timeout: int = 25) -> None:
    for url, name in ((grafana + "/api/health", "Grafana"),
                      (gateway + "/healthz", "gateway")):
        for _ in range(timeout):
            try:
                urllib.request.urlopen(url, timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            sys.exit(f"{name} not reachable at {url} — start it first "
                     f"(make observe / make serve, and `ollama serve`).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grafana", default="http://localhost:3000")
    ap.add_argument("--gateway", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-load", action="store_true",
                    help="skip load generation; screenshot the current state")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("Playwright not installed. Run:\n"
                 "  pip install playwright && playwright install chromium")

    OUT.mkdir(exist_ok=True)
    _wait_healthy(args.grafana, args.gateway)

    # Drive load so the time-series panels are populated; capture near the end
    # while traffic is still flowing.
    load_proc = None
    if not args.no_load:
        load_proc = subprocess.Popen([
            sys.executable, str(HERE / "loadgen.py"),
            "--gateway", args.gateway, "--model", args.model,
            "--duration", str(args.duration), "--concurrency", str(args.concurrency),
        ])
        lead = max(20, args.duration - 10)
        print(f"generating load; capturing in {lead}s...")
        time.sleep(lead)

    rng = f"from=now-15m&to=now&theme={args.theme}"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:
                sys.exit(f"Could not launch Chromium ({e}).\n"
                         "Run: playwright install chromium")
            # device_scale_factor=2 -> crisp retina-quality PNGs for the writeup.
            ctx = browser.new_context(device_scale_factor=2)

            # Full board (kiosk hides Grafana's nav chrome).
            page = ctx.new_page()
            page.set_viewport_size({"width": 1400, "height": 900})
            page.goto(f"{args.grafana}{DASH_FULL}?{rng}&kiosk", wait_until="load")
            page.wait_for_timeout(4000)  # let panels fetch + draw
            page.screenshot(path=str(OUT / "dashboard-full.png"), full_page=True)
            print("  saved dashboard-full.png")

            # Each panel via the d-solo single-panel viewer.
            page.set_viewport_size({"width": 1100, "height": 460})
            for pid in PANELS:
                page.goto(f"{args.grafana}{DASH_SOLO}?panelId={pid}&{rng}",
                          wait_until="load")
                page.wait_for_timeout(2500)
                page.screenshot(path=str(OUT / f"panel-{pid}.png"))
                print(f"  saved panel-{pid}.png")

            browser.close()
    finally:
        if load_proc:
            load_proc.wait()

    print(f"done — screenshots in {OUT}")


if __name__ == "__main__":
    main()
