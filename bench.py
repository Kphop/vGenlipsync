#!/usr/bin/env python3
"""Measure what one LatentSync clip actually costs on a RunPod endpoint.

Runs the endpoint N times, reads the telemetry each response carries, and prints
the numbers that decide whether this deployment is worth keeping: warm vs cold
cost per clip, the realtime factor needed to project other clip lengths, and
peak VRAM so you know whether a cheaper GPU tier would survive.

Runs on your machine, not in the container.

    set RUNPOD_API_KEY=...
    set RUNPOD_ENDPOINT_ID=...
    python bench.py --n 5

By default it uses the demo assets baked into the image, so there is no media to
host and no download variance in the timings.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import statistics
import sys
import time

import requests

API_BASE = "https://api.runpod.ai/v2"
POLL_INTERVAL_SEC = 3
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

# Published per-run prices for hosted LatentSync, for comparison. Verify before
# relying on them; provider pricing moves.
ALTERNATIVES = (("WaveSpeedAI", 0.05), ("Replicate", 0.10))

# RunPod serverless flex rates in $/hr, from runpod.io/pricing as of 2026-10.
# Matched as case-insensitive substrings of the GPU name the handler reports,
# so order matters: "l40s" must be tried before "l4".
#
# This table exists because RunPod assigns whatever card in the selected tier
# is free, which is not necessarily the one you asked for -- a run requested on
# a 4090 came back on an A5000, and pricing it at the 4090 rate overstated cost
# by 59%. Trust the reported GPU over your own assumption.
GPU_HOURLY = (
    ("h100", 4.79),
    ("a100", 2.72),
    ("l40s", 1.75),
    ("l40", 1.75),
    ("rtx 6000 ada", 1.75),
    ("a6000", 1.22),
    ("a40", 1.22),
    ("4090", 1.10),
    ("a5000", 0.69),
    ("3090", 0.69),
    ("l4", 0.69),
    ("a4000", 0.58),
    ("rtx 4000", 0.58),
)


def resolve_rate(gpu_name: str | None, explicit: float | None) -> tuple[float, str]:
    """Pick the $/hr rate to price with, preferring the detected GPU's own rate."""
    detected = None
    if gpu_name:
        low = gpu_name.lower()
        for key, price in GPU_HOURLY:
            if key in low:
                detected = price
                break

    if detected is None:
        rate = explicit if explicit is not None else 0.69
        how = (
            f"from --usd-per-hour (GPU {gpu_name or 'unknown'} not in the rate table)"
            if explicit is not None
            else f"DEFAULT GUESS -- {gpu_name or 'unknown GPU'} not in the rate table"
        )
        return rate, how

    if explicit is not None and abs(explicit - detected) > 0.001:
        print()
        print(
            f"  !! You passed --usd-per-hour {explicit:.2f} but RunPod ran this on a\n"
            f"     {gpu_name}, which bills at ${detected:.2f}/hr. Pricing at\n"
            f"     ${detected:.2f}/hr instead. RunPod assigns any free card in the\n"
            f"     tier, so the card you asked for is not always the card you get."
        )
        return detected, f"detected from {gpu_name}, overriding --usd-per-hour"

    return detected, f"detected from {gpu_name}"


def build_payload(args: argparse.Namespace) -> dict:
    # Off by default: a base64 mp4 inflates the response and the encode step
    # lands in the timings we are trying to measure. Turn it on with --save-video
    # when you want to judge output quality rather than cost.
    payload: dict = {"return_video": bool(args.save_video)}
    if args.video_url:
        payload["video_url"] = args.video_url
    else:
        payload["video_path"] = args.video_path
    if args.audio_url:
        payload["audio_url"] = args.audio_url
    else:
        payload["audio_path"] = args.audio_path
    if args.steps:
        payload["inference_steps"] = args.steps

    body: dict = {"input": payload}
    if args.exec_timeout:
        # Overrides the endpoint's Execution Timeout for this job only. A cold
        # request pays image pull plus a ~5 GB model load before inference even
        # starts, which can blow a console default that looks generous.
        body["policy"] = {"executionTimeout": int(args.exec_timeout * 1000)}
    return body


def submit(session: requests.Session, endpoint: str, payload: dict) -> str:
    resp = session.post(f"{API_BASE}/{endpoint}/run", json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    if "id" not in body:
        raise RuntimeError(f"unexpected /run response: {body}")
    return body["id"]


def wait(session: requests.Session, endpoint: str, job_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        resp = session.get(f"{API_BASE}/{endpoint}/status/{job_id}", timeout=60)
        resp.raise_for_status()
        body = resp.json()
        status = body.get("status")
        if status in TERMINAL:
            return body
        if time.monotonic() > deadline:
            raise TimeoutError(f"job {job_id} still {status} after {timeout:.0f}s")
        time.sleep(POLL_INTERVAL_SEC)


def save_video(out: dict, dest: pathlib.Path) -> str | None:
    """Write the returned mp4 to disk and drop the base64 from the result.

    Dropping it matters: the raw payload is kept for --out, and a few megabytes
    of base64 per run would otherwise end up in that JSON file.
    """
    encoded = out.pop("video_base64", None)
    if not encoded:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(encoded))
    return str(dest)


def run_once(session, endpoint, payload, timeout, save_path=None) -> dict:
    wall_start = time.monotonic()
    job_id = submit(session, endpoint, payload)
    body = wait(session, endpoint, job_id, timeout)
    wall = time.monotonic() - wall_start

    out = body.get("output") or {}
    if body.get("status") != "COMPLETED" or "error" in out:
        return {
            "ok": False,
            "job_id": job_id,
            "status": body.get("status"),
            "error": out.get("error") or body.get("error") or "unknown",
            "error_type": out.get("error_type"),
            "vram": out.get("vram"),
            "traceback": out.get("traceback"),
        }

    saved = save_video(out, save_path) if save_path else None

    return {
        "saved": saved,
        "ok": True,
        "job_id": job_id,
        "wall_sec": wall,
        # RunPod's own numbers are authoritative for billing; our in-handler
        # timings explain where that time went.
        "runpod_execution_sec": (body.get("executionTime") or 0) / 1000.0,
        "runpod_delay_sec": (body.get("delayTime") or 0) / 1000.0,
        "cold": out.get("worker", {}).get("cold_start"),
        "ready_after_sec": out.get("worker", {}).get("ready_after_sec"),
        "model_build_sec": out.get("worker", {}).get("model_build_sec"),
        "inference_sec": out.get("timings_sec", {}).get("inference"),
        "total_sec": out.get("timings_sec", {}).get("total"),
        "rtf": out.get("media", {}).get("realtime_factor"),
        "audio_sec": out.get("media", {}).get("audio_sec"),
        "peak_vram_gb": out.get("vram", {}).get("peak_reserved_gb"),
        "total_vram_gb": out.get("vram", {}).get("total_gb"),
        "gpu": out.get("vram", {}).get("gpu"),
        "handler_usd": out.get("cost", {}).get("usd_this_request"),
        "raw": out,
    }


def fmt(value, suffix="", width=0, nd=2):
    text = "-" if value is None else f"{value:.{nd}f}{suffix}"
    return text.rjust(width) if width else text


def report(results: list[dict], usd_per_hour: float | None, idle_sec: float) -> None:
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    gpu_name = ok[0].get("gpu") if ok else None
    usd_per_hour, rate_source = resolve_rate(gpu_name, usd_per_hour)
    usd_per_sec = usd_per_hour / 3600.0

    print()
    print("Run  Cold  Inference     Total   RunPod exec       RTF    PeakVRAM      $/clip")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        if not r["ok"]:
            print(f"{i:>3}  FAILED  {r.get('status')}: {str(r.get('error'))[:50]}")
            continue
        billed = r["runpod_execution_sec"] or r["wall_sec"]
        print(
            f"{i:>3}"
            f"{'  yes' if r['cold'] else '   no':>6}"
            f"{fmt(r['inference_sec'], 's', 11)}"
            f"{fmt(r['total_sec'], 's', 10)}"
            f"{fmt(billed, 's', 14)}"
            f"{fmt(r['rtf'], 'x', 10)}"
            f"{fmt(r['peak_vram_gb'], 'GB', 12)}"
            f"   ${billed * usd_per_sec:.4f}"
        )

    if not ok:
        print(f"\nAll {len(results)} runs failed. Nothing to price.")
        return

    warm = [r for r in ok if not r["cold"]]
    cold = [r for r in ok if r["cold"]]
    billed_of = lambda r: r["runpod_execution_sec"] or r["wall_sec"]  # noqa: E731

    print()
    print(f"Summary  (n={len(ok)} ok, {len(failed)} failed)")
    print(f"  GPU                 {gpu_name or 'unknown'}  @ ${usd_per_hour:.2f}/hr")
    print(f"  Rate source         {rate_source}")

    basis = warm or ok
    billed_times = sorted(billed_of(r) for r in basis)
    median_billed = statistics.median(billed_times)
    label = "warm" if warm else "cold (no warm runs yet)"
    print(f"  Billed per clip     median {median_billed:.1f}s  ({label})")
    if len(billed_times) >= 2:
        print(f"                      min {billed_times[0]:.1f}s  max {billed_times[-1]:.1f}s")

    warm_cost = median_billed * usd_per_sec
    print(f"  Cost per clip       ${warm_cost:.4f}")

    # RunPod bills executionTime, which spans more than the handler: returning a
    # large result keeps the worker busy uploading after the handler returns, and
    # that time is chargeable but invisible to our own timings. The gap exposes it.
    gaps = [
        billed_of(r) - r["total_sec"]
        for r in basis
        if r.get("total_sec") and r.get("runpod_execution_sec")
    ]
    if gaps:
        gap = statistics.median(gaps)
        encodes = [
            r["raw"].get("timings_sec", {}).get("encode")
            for r in basis
            if r.get("raw")
        ]
        enc = statistics.median([e for e in encodes if e]) if any(encodes) else 0.0
        print(
            f"  Platform overhead   {gap:.1f}s billed outside the handler"
            f"  (${gap * usd_per_sec:.4f}/clip)"
        )
        if enc:
            print(f"  of which base64     {enc:.1f}s encode inside the handler")
    print(f"  + idle timeout      ${(median_billed + idle_sec) * usd_per_sec:.4f}  ({idle_sec:.0f}s idle)")

    if cold:
        cold_billed = statistics.median(billed_of(r) for r in cold)
        print(
            f"  Cold start          ${cold_billed * usd_per_sec:.4f}/clip"
            f"  (model load {fmt(cold[0]['model_build_sec'], 's')},"
            f" ready after {fmt(cold[0]['ready_after_sec'], 's')})"
        )

    rtfs = [r["rtf"] for r in basis if r["rtf"]]
    if rtfs:
        rtf = statistics.median(rtfs)
        per_min = rtf * 60 * usd_per_sec
        print(f"  Realtime factor     {rtf:.2f}x  ->  1 min of audio ~ {rtf * 60:.0f}s ~ ${per_min:.4f}")

    peak = max((r["peak_vram_gb"] for r in ok if r["peak_vram_gb"]), default=None)
    total_vram = ok[0].get("total_vram_gb")
    if peak and total_vram:
        headroom = total_vram - peak
        if headroom > 6:
            verdict = "a smaller tier may fit -- try it"
        elif headroom < 3:
            verdict = f"only {headroom:.1f} GB spare -- OOM risk, size up"
        else:
            verdict = "this tier is right-sized"
        print(f"  Peak VRAM           {peak:.1f} GB of {total_vram:.1f} GB  ({verdict})")

    saved = [r["saved"] for r in ok if r.get("saved")]
    if saved:
        total_mb = sum(pathlib.Path(s).stat().st_size for s in saved) / 1024**2
        print(f"  Saved videos        {len(saved)} file(s), {total_mb:.1f} MB total")
        for s in saved:
            print(f"                        {s}")
        print("                      (timings include base64 encode/transfer)")

    audio_sec = ok[0].get("audio_sec")
    print()
    # Label the basis honestly: with no warm run yet these numbers carry the
    # cold-start cost and overstate steady-state spend.
    print(f"Monthly projection  ({label}, {audio_sec or '?'}s clips)")
    for volume in (100, 1_000, 10_000):
        print(f"  {volume:>6,} clips/mo    ${warm_cost * volume:>10,.2f}")

    print()
    print(f"vs pay-per-run hosted LatentSync  ({audio_sec or '?'}s clip)")
    print(f"  {'RunPod (this)':<16} ${warm_cost:.4f}")
    for name, price in ALTERNATIVES:
        if not warm_cost:
            verdict = "n/a"
        elif price > warm_cost:
            verdict = f"{price / warm_cost:.2f}x more expensive than RunPod"
        else:
            verdict = f"{warm_cost / price:.2f}x CHEAPER than RunPod"
        print(f"  {name:<16} ${price:.4f}   {verdict}")
    print()
    print(
        "  Note: RunPod wins per clip but carries the image build, the ~17 GB\n"
        "  pull and the cold-start tail. At low volume that overhead is the\n"
        "  real cost; compare against your own time, not just these numbers."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3, help="number of runs (default 3)")
    p.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    p.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    p.add_argument("--usd-per-hour", type=float, default=None,
                   help="override the $/hr used for cost math. By default the rate is "
                        "looked up from the GPU the worker actually reports, which is "
                        "more reliable than assuming the card you selected.")
    p.add_argument("--idle-timeout", type=float, default=5.0,
                   help="endpoint idle timeout, included in the full per-clip price")
    p.add_argument("--video-path", default="assets/demo1_video.mp4",
                   help="path to a file baked into the image (default: repo demo)")
    p.add_argument("--audio-path", default="assets/demo1_audio.wav")
    p.add_argument("--video-url", help="use a remote video instead of the baked asset")
    p.add_argument("--audio-url")
    p.add_argument("--steps", type=int, help="override inference_steps")
    p.add_argument("--save-video", action="store_true",
                   help="return each mp4 and write it to --save-dir, to judge output "
                        "quality. Adds base64 encode/transfer time to the measurement, "
                        "so leave it off for the cost numbers you intend to quote.")
    p.add_argument("--save-dir", default="outputs",
                   help="directory for --save-video mp4s (default: outputs/)")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="seconds to wait between runs; exceed the idle timeout to force cold starts")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="how long this client waits before giving up on a job")
    p.add_argument("--exec-timeout", type=float, default=None,
                   help="override the endpoint's Execution Timeout for these jobs, "
                        "in seconds (sent as a per-request policy). Use this when runs "
                        "fail with 'executionTimeout exceeded'.")
    p.add_argument("--out", help="write raw results to this JSON file")
    args = p.parse_args()

    if not args.endpoint_id or not args.api_key:
        p.error("set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY, or pass --endpoint-id/--api-key")

    payload = build_payload(args)
    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    )

    print(f"endpoint {args.endpoint_id}  n={args.n}  payload={json.dumps(payload['input'])}")
    results = []
    for i in range(args.n):
        if i and args.sleep:
            print(f"  sleeping {args.sleep:.0f}s before run {i + 1}")
            time.sleep(args.sleep)
        print(f"  run {i + 1}/{args.n} ...", end="", flush=True)
        save_path = (
            pathlib.Path(args.save_dir) / f"run-{i + 1:02d}.mp4" if args.save_video else None
        )
        try:
            r = run_once(session, args.endpoint_id, payload, args.timeout, save_path)
        except Exception as exc:
            r = {"ok": False, "status": "CLIENT_ERROR", "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        if r["ok"]:
            msg = f" ok  {r['runpod_execution_sec']:.1f}s billed"
            if r.get("saved"):
                msg += f"  -> {r['saved']}"
            print(msg)
        else:
            print(f" FAILED  {str(r.get('error'))[:70]}")
            if r.get("error_type"):
                print(f"           error_type={r['error_type']}")
            if r.get("vram"):
                print(f"           vram={r['vram']}")

    report(results, args.usd_per_hour, args.idle_timeout)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"raw results -> {args.out}")

    return 0 if any(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
