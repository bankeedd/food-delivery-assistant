"""
Web UI for the Mini Food Delivery Assistant.
Uses only the Python standard library — no installs needed.

Run:   python app.py          (opens http://localhost:8000 in your browser)
       python app.py 8080     (custom port)
"""

import json
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import delivery_assistant as da

STATIC = Path(__file__).parent / "static"


def to_json(obj):
    """Tuples -> lists, recursively, so the result is JSON-safe."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if obj == float("inf"):
        return None
    return obj


def validate_map(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("map must be a non-empty list of strings")
    width = len(rows[0])
    for row in rows:
        if not isinstance(row, str) or len(row) != width:
            raise ValueError("all map rows must be strings of equal length")
        if set(row) - set("SC.~#"):
            raise ValueError("map may only contain S C . ~ #")
    if sum(r.count("S") for r in rows) != 1 or sum(r.count("C") for r in rows) != 1:
        raise ValueError("map needs exactly one S and one C")
    return rows


def handle_solve(body):
    rows = validate_map(body.get("map", da.CITY_MAP))
    cell = body.get("alert_cell")
    return da.solve(
        rows=rows,
        alert=bool(body.get("alert", True)),
        prior=min(max(float(body.get("prior", da.P_BLOCKED_PRIOR)), 0.001), 0.999),
        alert_cell=tuple(cell) if cell else None,
        p_a_b=min(max(float(body.get("p_a_b", da.P_ALERT_IF_BLOCKED)), 0.01), 0.99),
        p_a_nb=min(max(float(body.get("p_a_nb", da.P_ALERT_IF_CLEAR)), 0.01), 0.99),
        jam_wait=max(float(body.get("jam_wait", da.JAM_WAIT)), 0),
        gamma=min(max(float(body.get("gamma", da.GAMMA)), 0), 0.99),
    )


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):   # keep the console quiet during the demo
        pass

    def _send_json(self, code, payload):
        data = json.dumps(to_json(payload)).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/defaults":
            return self._send_json(200, {
                "map": da.CITY_MAP, "prior": da.P_BLOCKED_PRIOR,
                "p_a_b": da.P_ALERT_IF_BLOCKED, "p_a_nb": da.P_ALERT_IF_CLEAR,
                "jam_wait": da.JAM_WAIT, "gamma": da.GAMMA})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/solve":
            return self._send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            self._send_json(200, handle_solve(body))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Food Delivery Assistant running at {url}  (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
