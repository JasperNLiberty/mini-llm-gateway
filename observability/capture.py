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
PANELS = list(range(1, 13))  # panel ids 1..12 (10 core + 2 reasoning)


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
    ap.add_argument("--reasoning", action="store_true",
                    help="drive reasoning traffic (/ollama/think/stream) to light up "
                         "the thinking-token panels")
    ap.add_argument("--from", dest="frm", default=None,
                    help="Grafana time range start: epoch ms or relative (e.g. now-15m). "
                         "Default: now-15m. Pass an absolute epoch-ms window to pin a "
                         "screenshot to one experiment's run.")
    ap.add_argument("--to", default=None, help="time range end (epoch ms or 'now'). Default: now")
    ap.add_argument("--out-dir", default=None,
                    help="directory to write PNGs into (default observability/screenshots/)")
    ap.add_argument("--prefix", default="", help="filename prefix, e.g. '01-' or 'reasoning-'")
    ap.add_argument("--panels", default=None,
                    help="comma-separated panel ids to capture (default: all 1..12)")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("Playwright not installed. Run:\n"
                 "  pip install playwright && playwright install chromium")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    frm = args.frm or "now-15m"
    to = args.to or "now"
    prefix = args.prefix
    panels = ([int(p) for p in args.panels.split(",")] if args.panels else PANELS)
    _wait_healthy(args.grafana, args.gateway)

    # Drive load so the time-series panels are populated; capture near the end
    # while traffic is still flowing.
    load_proc = None
    if not args.no_load:
        load_cmd = [
            sys.executable, str(HERE / "loadgen.py"),
            "--gateway", args.gateway, "--model", args.model,
            "--duration", str(args.duration), "--concurrency", str(args.concurrency),
        ]
        if args.reasoning:
            load_cmd.append("--reasoning")
        load_proc = subprocess.Popen(load_cmd)
        lead = max(20, args.duration - 10)
        print(f"generating load; capturing in {lead}s...")
        time.sleep(lead)

    rng = f"from={frm}&to={to}&theme={args.theme}"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:
                sys.exit(f"Could not launch Chromium ({e}).\n"
                         "Run: playwright install chromium")
            # device_scale_factor=2 -> crisp retina-quality PNGs for the writeup.
            ctx = browser.new_context(device_scale_factor=2)

            # Full board (kiosk hides Grafana's nav chrome). Grafana lazy-loads
            # panels as they scroll into view, so a plain full_page shot comes
            # out blank. Use a tall viewport AND wheel-scroll through the page to
            # force every panel to render, then scroll back to top and capture.
            page = ctx.new_page()
            page.set_viewport_size({"width": 1500, "height": 2400})
            page.goto(f"{args.grafana}{DASH_FULL}?{rng}&kiosk", wait_until="load")
            page.wait_for_timeout(2500)
            for _ in range(8):
                page.mouse.wheel(0, 500)
                page.wait_for_timeout(350)
            page.mouse.wheel(0, -6000)   # back to top
            page.wait_for_timeout(2500)  # let charts settle
            page.screenshot(path=str(out_dir / f"{prefix}dashboard-full.png"), full_page=True)
            print(f"  saved {prefix}dashboard-full.png")

            # Each panel via the d-solo single-panel viewer.
            page.set_viewport_size({"width": 1100, "height": 460})
            for pid in panels:
                page.goto(f"{args.grafana}{DASH_SOLO}?panelId={pid}&{rng}",
                          wait_until="load")
                page.wait_for_timeout(2500)
                page.screenshot(path=str(out_dir / f"{prefix}panel-{pid}.png"))
                print(f"  saved {prefix}panel-{pid}.png")

            browser.close()
    finally:
        if load_proc:
            load_proc.wait()

    print(f"done — screenshots in {out_dir}")


if __name__ == "__main__":
    main()
