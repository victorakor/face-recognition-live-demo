"""Flask app for the live face-recognition demo.

There is no login, no database and no server-side camera. A visitor's browser captures
frames from their own webcam, POSTs them to /api/detect, and draws the returned boxes as
an overlay. Frames are decoded in memory, run through the model, and discarded -- nothing
is written to disk.
"""

from __future__ import annotations

import base64
import binascii
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from recognizer import DEFAULT_MODE, DEFAULT_THRESHOLD, MODES, get_recognizer, is_ready, warmup

# Uploaded frames are ~40-120 KB of JPEG; 6 MB leaves room for a full-resolution
# still from the image-upload fallback without letting anyone stream junk at us.
MAX_CONTENT_LENGTH = 6 * 1024 * 1024

# Inference is serialised inside the recogniser anyway, so admitting a deep queue only
# adds latency. Extra callers get a fast 503 and the client simply skips that frame.
MAX_IN_FLIGHT = int(os.getenv("MAX_IN_FLIGHT", "3"))

# Per-IP token bucket. Clients target ~6 fps, so this is generous for a real visitor and
# still caps what a single host can spend of the container's CPU.
RATE_LIMIT_BURST = float(os.getenv("RATE_LIMIT_BURST", "24"))
RATE_LIMIT_PER_SEC = float(os.getenv("RATE_LIMIT_PER_SEC", "12"))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["JSON_SORT_KEYS"] = False

_in_flight = threading.BoundedSemaphore(MAX_IN_FLIGHT)
_buckets: dict[str, list[float]] = {}
_buckets_lock = threading.Lock()


def client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_limited(ip: str) -> bool:
    now = time.monotonic()
    with _buckets_lock:
        # Bounded memory: drop buckets that have been idle for a minute. Cheap because
        # the dict only ever holds concurrent visitors.
        if len(_buckets) > 512:
            for stale_ip in [k for k, (_, seen) in _buckets.items() if now - seen > 60]:
                del _buckets[stale_ip]

        tokens, last_seen = _buckets.get(ip, [RATE_LIMIT_BURST, now])
        tokens = min(RATE_LIMIT_BURST, tokens + (now - last_seen) * RATE_LIMIT_PER_SEC)
        if tokens < 1.0:
            _buckets[ip] = [tokens, now]
            return True
        _buckets[ip] = [tokens - 1.0, now]
        return False


def read_frame_bytes() -> bytes:
    """Accept either a raw JPEG body or a JSON payload carrying a data URL."""
    content_type = (request.content_type or "").split(";")[0].strip().lower()

    if content_type == "application/json":
        payload = request.get_json(silent=True) or {}
        encoded = payload.get("image", "")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("json body must contain an 'image' string")
        if "," in encoded and encoded.lstrip().startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        try:
            return base64.b64decode(encoded, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image is not valid base64: {exc}") from exc

    if request.files:
        return next(iter(request.files.values())).read()

    return request.get_data(cache=False)


def decode_frame(raw: bytes) -> np.ndarray:
    if not raw:
        raise ValueError("empty request body")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode image (expected JPEG or PNG)")
    return frame


def requested_threshold() -> float:
    raw = request.args.get("threshold") or request.headers.get("X-Threshold")
    if raw is None and (request.content_type or "").startswith("application/json"):
        payload = request.get_json(silent=True) or {}
        raw = payload.get("threshold")
    if raw is None:
        return DEFAULT_THRESHOLD
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD


def requested_mode() -> str:
    """Which pipeline to run: "recognition" (identify faces) or "detection" (classes only).

    Read from the same three places as the threshold. An unrecognised value falls back to
    the default rather than erroring -- a typo in a query string should not break the demo.
    """
    raw = request.args.get("mode") or request.headers.get("X-Mode")
    if raw is None and (request.content_type or "").startswith("application/json"):
        payload = request.get_json(silent=True) or {}
        raw = payload.get("mode")
    if not isinstance(raw, str):
        return DEFAULT_MODE
    mode = raw.strip().lower()
    return mode if mode in MODES else DEFAULT_MODE


@app.route("/")
def index() -> str:
    return render_template(
        "index.html", default_threshold=DEFAULT_THRESHOLD, default_mode=DEFAULT_MODE
    )


@app.route("/api/health")
def health() -> Response:
    # Deliberately does not force a model load, so platform health checks pass while the
    # weights are still coming up on the warm-up thread.
    return jsonify({"ok": True, "ready": is_ready()})


@app.route("/api/info")
def info() -> Response:
    if not is_ready():
        return jsonify({"ready": False})

    recognizer = get_recognizer()
    return jsonify(
        {
            "ready": True,
            "model": recognizer.info(),
            "evaluation": {
                "samples": 55,
                "rocAuc": 0.9857,
                "distanceThreshold": 0.402,
                "precision": 0.8333,
                "recall": 1.0,
                "f1": 0.9091,
                "accuracy": 0.9273,
                "confusion": {"tp": 20, "fn": 0, "fp": 4, "tn": 31},
            },
        }
    )


@app.route("/api/detect", methods=["POST"])
def detect() -> tuple[Response, int] | Response:
    if rate_limited(client_ip()):
        return jsonify({"error": "rate limited", "retry": True}), 429

    try:
        raw = read_frame_bytes()
        frame = decode_frame(raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not _in_flight.acquire(blocking=False):
        return jsonify({"error": "server busy", "retry": True}), 503
    try:
        result = get_recognizer().detect(
            frame, threshold=requested_threshold(), mode=requested_mode()
        )
    except Exception as exc:  # pragma: no cover - surfaced to the client for debugging
        app.logger.exception("detection failed")
        return jsonify({"error": f"detection failed: {exc}"}), 500
    finally:
        _in_flight.release()

    return jsonify(result)


@app.errorhandler(413)
def too_large(_error) -> tuple[Response, int]:
    return jsonify({"error": f"image too large (max {MAX_CONTENT_LENGTH // (1024 * 1024)} MB)"}), 413


@app.after_request
def security_headers(response: Response) -> Response:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # getUserMedia needs an explicit grant when the page is framed, which is how
    # Hugging Face Spaces renders it.
    response.headers.setdefault("Permissions-Policy", "camera=(self)")
    return response


# Start loading the models immediately, on a background thread, whether we were started by
# gunicorn or run directly.
warmup()


if __name__ == "__main__":
    # Hugging Face Spaces defaults to 7860; Render injects PORT.
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
