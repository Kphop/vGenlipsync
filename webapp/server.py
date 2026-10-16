#!/usr/bin/env python3
"""Local web UI for the LatentSync endpoint.

Why a server and not just an HTML file: the RunPod API key must never reach the
browser. A page calling api.runpod.ai directly would ship a key that can spend
money and control the whole RunPod account to anyone who opens view-source.
This process holds the key; the browser only ever talks to localhost.

    set RUNPOD_API_KEY=...
    set RUNPOD_ENDPOINT_ID=...
    python webapp/server.py

Binds to 127.0.0.1 only. Do not expose it to a network without adding auth --
every request it forwards costs real money.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

# The Windows console defaults to cp1252, which cannot encode Thai -- every
# print() below would raise UnicodeEncodeError and kill the server before it
# serves anything. errors="replace" keeps a console with a limited font from
# crashing it too.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parent
OUTPUT_DIR = PROJECT / "outputs" / "web"

API_BASE = "https://api.runpod.ai/v2"
API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()

# RunPod rejects /run bodies over ~10 MB. Base64 inflates by 4/3, so cap the
# decoded total below that and fail here with a clear message rather than
# letting RunPod reject it with an opaque one.
MAX_UPLOAD_BYTES = 6 * 1024**2
MAX_REQUEST_BYTES = 32 * 1024**2

# Job runs take minutes; the endpoint's own Execution Timeout still applies.
EXEC_TIMEOUT_SEC = int(os.environ.get("EXEC_TIMEOUT_SEC", "1800"))

JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def rp_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    return s


class Handler(BaseHTTPRequestHandler):
    server_version = "latentsync-webui"

    # ---------------- helpers ----------------
    def _send(self, code: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("empty body")
        if length > MAX_REQUEST_BYTES:
            raise ValueError(f"request body too large ({length / 1024**2:.1f} MB)")
        return json.loads(self.rfile.read(length))

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"  {self.address_string()} - {fmt % args}\n")

    # ---------------- routing ----------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            html = (HERE / "index.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")

        if path == "/api/config":
            return self._json(200, {
                "configured": bool(API_KEY and ENDPOINT_ID),
                "endpoint_id": ENDPOINT_ID,
                "max_upload_mb": round(MAX_UPLOAD_BYTES / 1024**2, 1),
            })

        if path.startswith("/api/jobs/"):
            return self._status(path.rsplit("/", 1)[-1])

        if path.startswith("/outputs/"):
            return self._serve_output(path.rsplit("/", 1)[-1])

        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/jobs":
            return self._json(404, {"error": "not found"})
        try:
            return self._submit(self._read_json())
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    # ---------------- endpoints ----------------
    def _submit(self, body: dict) -> None:
        if not (API_KEY and ENDPOINT_ID):
            return self._json(400, {
                "error": "ยังไม่ได้ตั้ง RUNPOD_API_KEY หรือ RUNPOD_ENDPOINT_ID "
                         "ปิดเซิร์ฟเวอร์ ตั้งค่า แล้วเปิดใหม่"
            })

        visual_b64 = (body.get("visual_b64") or "").strip()
        audio_b64 = (body.get("audio_b64") or "").strip()
        if not visual_b64 or not audio_b64:
            raise ValueError("ต้องมีทั้งไฟล์ภาพ/วิดีโอ และไฟล์เสียง")

        for label, blob in (("ภาพ/วิดีโอ", visual_b64), ("เสียง", audio_b64)):
            decoded = len(blob) * 3 // 4
            if decoded > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"ไฟล์{label}ใหญ่ {decoded / 1024**2:.1f} MB "
                    f"เกินขีดจำกัด {MAX_UPLOAD_BYTES / 1024**2:.0f} MB "
                    "(RunPod จำกัด request ที่ ~10 MB) ลองตัดคลิปให้สั้นลง"
                )

        # The handler takes a still image or a video for the visual input and
        # loops a still to match the audio length by itself.
        visual_key = "image_base64" if body.get("visual_is_image") else "video_base64"

        payload = {
            "input": {
                visual_key: visual_b64,
                "audio_base64": audio_b64,
                "return_video": True,
                "inference_steps": int(body.get("steps") or 20),
            },
            "policy": {"executionTimeout": EXEC_TIMEOUT_SEC * 1000},
        }

        resp = rp_session().post(f"{API_BASE}/{ENDPOINT_ID}/run", json=payload, timeout=120)
        if resp.status_code >= 400:
            return self._json(resp.status_code, {"error": f"RunPod ตอบ {resp.status_code}: {resp.text[:400]}"})

        job_id = resp.json().get("id")
        if not job_id:
            return self._json(502, {"error": f"RunPod ไม่ได้คืน job id: {resp.text[:400]}"})
        print(f"  submitted job {job_id} (steps={payload['input']['inference_steps']})")
        return self._json(200, {"id": job_id})

    def _status(self, job_id: str) -> None:
        if not JOB_ID_RE.match(job_id):
            return self._json(400, {"error": "job id ไม่ถูกต้อง"})

        resp = rp_session().get(f"{API_BASE}/{ENDPOINT_ID}/status/{job_id}", timeout=60)
        if resp.status_code >= 400:
            return self._json(resp.status_code, {"error": f"RunPod ตอบ {resp.status_code}"})

        body = resp.json()
        out = body.get("output") or {}
        result = {
            "status": body.get("status"),
            "delay_sec": round((body.get("delayTime") or 0) / 1000, 1),
            "execution_sec": round((body.get("executionTime") or 0) / 1000, 1),
        }

        if body.get("status") != "COMPLETED":
            if body.get("status") in ("FAILED", "CANCELLED", "TIMED_OUT"):
                result["error"] = out.get("error") or body.get("error") or body.get("status")
                result["error_type"] = out.get("error_type")
            return self._json(200, result)

        if "error" in out:
            result["error"] = out["error"]
            result["error_type"] = out.get("error_type")
            return self._json(200, result)

        # Decode once here and hand the browser a plain URL: a <video> tag
        # playing a multi-megabyte data: URI is slow and cannot be downloaded
        # with a normal link.
        encoded = out.pop("video_base64", None)
        if encoded:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            dest = OUTPUT_DIR / f"{job_id}.mp4"
            dest.write_bytes(base64.b64decode(encoded))
            result["video_url"] = f"/outputs/{dest.name}"
            result["video_bytes"] = dest.stat().st_size
            print(f"  job {job_id} -> {dest}")

        result["telemetry"] = out
        return self._json(200, result)

    def _serve_output(self, name: str) -> None:
        # Resolve and confirm containment so a crafted name cannot walk out of
        # the output directory.
        dest = (OUTPUT_DIR / name).resolve()
        if not dest.is_relative_to(OUTPUT_DIR.resolve()) or not dest.is_file():
            return self._json(404, {"error": "not found"})
        data = dest.read_bytes()
        self._send(200, data, "video/mp4", {
            "Content-Disposition": f'attachment; filename="{dest.name}"',
            "Accept-Ranges": "none",
        })


def main() -> int:
    port = int(os.environ.get("PORT", "8000"))

    if not (API_KEY and ENDPOINT_ID):
        print("!! RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID ยังไม่ได้ตั้ง")
        print("   หน้าเว็บจะเปิดได้แต่ส่งงานไม่ได้ ตั้งค่าแล้วเปิดใหม่:")
        print('   $env:RUNPOD_API_KEY="..."; $env:RUNPOD_ENDPOINT_ID="..."')
        print()

    url = f"http://127.0.0.1:{port}/"
    print(f"  LatentSync web UI -> {url}")
    print(f"  endpoint: {ENDPOINT_ID or '(ไม่ได้ตั้ง)'}")
    print(f"  outputs : {OUTPUT_DIR}")
    print("  กด Ctrl+C เพื่อปิด\n")

    # 127.0.0.1, not 0.0.0.0: anyone who can reach this port can spend money.
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  ปิดเซิร์ฟเวอร์")
    return 0


if __name__ == "__main__":
    sys.exit(main())
