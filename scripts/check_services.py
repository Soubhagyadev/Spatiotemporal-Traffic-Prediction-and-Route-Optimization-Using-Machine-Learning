"""Smoke-test the three Part 2 services against their running HTTP endpoints.

Usage
-----
    python scripts/check_services.py [--api-url ...] [--dashboard-url ...] [--web-url ...]

Checks API payload shapes, executes every Dash callback with realistic
control values, and fetches each Django page.  Exits non-zero if any check
fails, so it can be used as an acceptance test after starting the services.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import requests  # noqa: E402

TIMEOUT = 120
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}")


def get(url: str, **kwargs):
    return requests.get(url, timeout=TIMEOUT, **kwargs)


def post(url: str, payload: dict):
    return requests.post(url, json=payload, timeout=TIMEOUT)


# ----------------------------------------------------------------------
def check_api(base: str) -> None:
    print(f"\nAPI  {base}")
    try:
        health = get(f"{base}/health").json()
    except requests.RequestException as exc:
        record("api.reachable", False, str(exc))
        return
    record("api.reachable", True, f"{health['dataset']['link_hours']:,} link-hours")
    record(
        "api.model_loaded",
        health["model"]["feature_count"] == 23 and len(health["model"]["models"]) == 2,
        f"{health['model']['models']}",
    )
    endpoints = [
        ("/api/meta", "dataset"),
        ("/api/links", "links"),
        ("/api/links/4616341", "link_id"),
        ("/api/links/4616341/history?granularity=hour", "points"),
        ("/api/traffic/over_time?granularity=day", "points"),
        ("/api/traffic/over_time?granularity=hour&link_id=4616341", "points"),
        ("/api/patterns/hourly", "points"),
        ("/api/patterns/weekday_hour", "points"),
        ("/api/congestion/segments?limit=5", "segments"),
        ("/api/congestion/boroughs", "boroughs"),
        ("/api/congestion/summary", "summary"),
        ("/api/model/metrics", "results"),
        ("/api/model/predictions?limit=10", "points"),
        ("/api/model/errors?limit=5", "segments"),
        ("/api/predict/latest/4616341", "predicted_speed_mph"),
    ]
    for path, key in endpoints:
        try:
            response = get(base + path)
            payload = response.json()
            ok = response.status_code == 200 and key in payload and payload[key] not in ([], {}, None)
            record(f"api{path.split('?')[0]}", ok, f"{response.status_code} key={key}")
        except (requests.RequestException, ValueError) as exc:
            record(f"api{path.split('?')[0]}", False, str(exc)[:60])

    prediction = post(
        f"{base}/api/predict", {"link_id": 4616341, "hour_ts": "2026-10-07 18:00:00"}
    ).json()
    speed = prediction.get("predicted_speed_mph")
    record(
        "api.predict",
        isinstance(speed, (int, float))
        and 0 < speed < 120
        and prediction.get("congestion") in {"High", "Medium", "Low"}
        and prediction.get("mode") in {"observed-lag", "forecast"},
        f"{speed} mph, {prediction.get('congestion')}, {prediction.get('mode')}, "
        f"lags {prediction.get('lags_observed')}/{prediction.get('lags_total')}",
    )
    horizon = post(
        f"{base}/api/predict/horizon",
        {"link_id": 4616341, "start_hour_ts": "2026-10-07 20:00:00", "steps": 5},
    ).json()
    record("api.predict_horizon", len(horizon.get("predictions", [])) == 5, "5 steps")
    batch = post(
        f"{base}/api/predict/batch",
        {"requests": [{"link_id": 4616341, "hour_ts": "2026-10-07 18:00:00"}]},
    ).json()
    record("api.predict_batch", batch.get("count") == 1, f"count={batch.get('count')}")
    bad = post(f"{base}/api/predict", {"link_id": 4616341, "hour_ts": "nonsense"})
    record("api.validation_error", bad.status_code == 400, f"{bad.status_code}")


# ----------------------------------------------------------------------
def _dash_outputs(spec: str) -> list[dict]:
    return [
        dict(zip(("id", "property"), part.rsplit(".", 1)))
        for part in spec.strip(".").split("...")
    ]


def _anchors(node, collected: list) -> None:
    """Collect every ``href`` in a Dash layout tree."""
    if isinstance(node, dict):
        if node.get("type") == "A":
            collected.append(node.get("props", {}).get("href"))
        for value in node.values():
            _anchors(value, collected)
    elif isinstance(node, list):
        for value in node:
            _anchors(value, collected)


def check_dashboard(base: str, web_url: str = "") -> None:
    print(f"\nDASHBOARD  {base}")
    try:
        layout = get(f"{base}/")
    except requests.RequestException as exc:
        record("dashboard.reachable", False, str(exc))
        return
    record("dashboard.reachable", layout.status_code == 200, f"{len(layout.text):,} bytes")
    dependencies = get(f"{base}/_dash-dependencies").json()
    record("dashboard.callbacks_registered", len(dependencies) == 4, f"{len(dependencies)}")

    anchors: list = []
    _anchors(get(f"{base}/_dash-layout").json(), anchors)
    record(
        "dashboard.link_to_web",
        bool(web_url) and web_url.rstrip("/") in [str(a).rstrip("/") for a in anchors if a],
        f"header links to {web_url}",
    )

    scenarios = [
        {"ctl-borough": "all", "ctl-link": "all", "ctl-granularity": "day",
         "ctl-model": "linear_regression"},
        {"ctl-borough": "Manhattan", "ctl-link": 4616341, "ctl-granularity": "hour",
         "ctl-model": "random_forest"},
    ]
    for index, scenario in enumerate(scenarios, start=1):
        for dependency in dependencies:
            outputs = _dash_outputs(dependency["output"])
            inputs = [
                {"id": item["id"], "property": item["property"], "value": scenario[item["id"]]}
                for item in dependency["inputs"]
            ]
            try:
                response = post(
                    f"{base}/_dash-update-component",
                    {
                        "output": dependency["output"],
                        "outputs": outputs,
                        "inputs": inputs,
                        "changedPropIds": [f"{inputs[0]['id']}.{inputs[0]['property']}"],
                    },
                )
            except requests.RequestException as exc:
                record(f"dashboard.callback[{index}].{outputs[0]['id']}", False, str(exc)[:50])
                continue
            body = response.json().get("response", {}) if response.status_code == 200 else {}
            filled = sum(
                1
                for out in outputs
                if body.get(out["id"], {}).get(out["property"]) not in (None, "", [], {})
            )
            record(
                f"dashboard.callback[{index}].{outputs[0]['id']}",
                response.status_code == 200 and filled >= len(outputs) - 1,
                f"{response.status_code} {filled}/{len(outputs)} outputs filled",
            )


# ----------------------------------------------------------------------
def check_web(base: str, dashboard_url: str = "") -> None:
    print(f"\nDJANGO  {base}")
    pages = [
        ("/", "NYC"),
        ("/segments/", "segment"),
        ("/segments/?borough=Manhattan", "Manhattan"),
        ("/segments/4616341/", "4616341"),
        ("/predict/", "predict"),
        ("/model/", "mae"),
        ("/about/", "719,046"),
    ]
    for path, marker in pages:
        try:
            response = get(base + path)
            ok = response.status_code == 200 and marker.lower() in response.text.lower()
            record(f"web{path}", ok, f"{response.status_code} marker={marker!r}")
        except requests.RequestException as exc:
            record(f"web{path}", False, str(exc)[:60])

    home = get(base + "/")
    record(
        "web.link_to_dashboard",
        bool(dashboard_url) and dashboard_url.rstrip("/") in home.text,
        f"navbar links to {dashboard_url}",
    )

    # The prediction form is the only state-changing page: post it for real,
    # including the CSRF token Django requires.
    try:
        session = requests.Session()
        form = session.get(f"{base}/predict/", timeout=TIMEOUT)
        token = session.cookies.get("csrftoken") or ""
        response = session.post(
            f"{base}/predict/",
            timeout=TIMEOUT,
            data={
                "csrfmiddlewaretoken": token,
                "link_id": "4616341",
                "hour_ts": "2026-10-07 20:00:00",
                "model": "linear_regression",
                "steps": "4",
            },
            headers={"Referer": f"{base}/predict/"},
        )
        text = response.text.lower()
        ok = (
            response.status_code == 200
            and "predicted speed" in text
            and "congestion" in text
            and form.status_code == 200
        )
        record("web/predict POST (CSRF)", ok, f"{response.status_code} with result rendered")
    except requests.RequestException as exc:
        record("web/predict POST (CSRF)", False, str(exc)[:60])

    try:
        missing = get(f"{base}/segments/999999/")
        record("web/segments unknown -> 404", missing.status_code == 404, f"{missing.status_code}")
    except requests.RequestException as exc:
        record("web/segments unknown -> 404", False, str(exc)[:60])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:5001")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8050")
    parser.add_argument("--web-url", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--skip-web", action="store_true")
    args = parser.parse_args()

    check_api(args.api_url)
    if not args.skip_dashboard:
        check_dashboard(args.dashboard_url, args.web_url)
    if not args.skip_web:
        check_web(args.web_url, args.dashboard_url)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} — {detail}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
