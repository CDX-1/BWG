"""Loopback JSON feed for the browser-side attitude view.

Streamlit can only repaint by rerunning on the server, which remounts the chart
and shows as a flicker. Instead the orientation panel is drawn in the browser
and polls this endpoint directly, so the page itself never has to rerun.

Bound to 127.0.0.1 only — nothing here is reachable off the machine.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def build_snapshot(link) -> dict:
    """Current attitude plus the rates needed to extrapolate between frames."""
    sample = link.latest()
    if sample is None:
        return {"live": False, "connected": link.is_running()}

    return {
        "live": link.is_live(),
        "connected": link.is_running(),
        # Seconds since this frame was captured, so the browser can predict
        # forward without needing our clock.
        "age_s": max(0.0, time.time() - sample.received_at),
        "roll": sample.roll,
        "pitch": sample.pitch,
        "yaw": sample.yaw,
        "gyro": [sample.gyro_x, sample.gyro_y, sample.gyro_z],
        "accel": [sample.accel_x, sample.accel_y, sample.accel_z],
        "rate": sample.gyro_magnitude,
        "tilt": sample.tilt_deg,
        "temp_c": sample.temp_c,
        "distance_cm": sample.distance_cm,
        "rate_hz": link.sample_rate_hz(),
        "frames": link.lines_seen,
    }


class _Handler(BaseHTTPRequestHandler):
    link = None   # set on the subclass created in start_server

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The panel runs in a srcdoc iframe, whose origin is "null".
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/orientation"):
            self._send(build_snapshot(self.link))
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path.startswith("/relevel"):
            self.link.reset_orientation()
            self._send({"ok": True})
        else:
            self._send({"error": "not found"}, status=404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def log_message(self, *args):
        pass   # polling at 20 Hz would otherwise flood stderr


def start_server(link) -> int:
    """Start the feed on an ephemeral loopback port and return that port."""
    handler = type("_BoundHandler", (_Handler,), {"link": link})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]
