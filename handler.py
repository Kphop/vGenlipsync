#!/usr/bin/env python3
"""RunPod serverless handler for LatentSync 1.6, instrumented for cost measurement.

The point of this endpoint is not to be a polished API -- it is to answer one
question with real numbers: what does one lip-synced clip actually cost on
RunPod? Every response therefore carries a timing breakdown, peak VRAM, and a
computed dollar figure alongside the video.

Two measurements matter most:

  * ``realtime_factor``   -- inference seconds per second of audio. This is the
    only number needed to project cost for clips of any other length.
  * ``vram.peak_reserved_gb`` -- whether upstream's stated 18 GB is real, and
    therefore whether a cheaper GPU tier would work.

Model loading happens once at import, outside the handler, so the GPU-seconds
spent on it are paid per worker rather than per request.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pathlib
import shutil
import subprocess
import threading
import time
import traceback
import uuid

import runpod
import torch

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
LATENTSYNC_ROOT = pathlib.Path(os.environ.get("LATENTSYNC_ROOT", "/opt/latentsync"))

# Upstream resolves DDIMScheduler.from_pretrained("configs"),
# FaceAnalysis(root="checkpoints/auxiliary") and config.data.mask_image_path as
# paths relative to the repo root, so the process must run from there.
os.chdir(LATENTSYNC_ROOT)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


UNET_CONFIG_PATH = os.environ.get("UNET_CONFIG_PATH", "configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = os.environ.get("INFERENCE_CKPT_PATH", "checkpoints/latentsync_unet.pt")

# Upstream's inference.sh defaults for 1.6.
DEFAULT_INFERENCE_STEPS = _env_int("INFERENCE_STEPS", 20)
DEFAULT_GUIDANCE_SCALE = _env_float("GUIDANCE_SCALE", 1.5)
ENABLE_DEEPCACHE = _env_bool("ENABLE_DEEPCACHE", True)
SEED = _env_int("SEED", 1247)

# $0.69/hr is the 24 GB flex tier (L4 / A5000 / RTX 3090), the cheapest class
# with enough VRAM for 1.6. Override when benchmarking a different GPU.
GPU_USD_PER_HOUR = _env_float("GPU_USD_PER_HOUR", 0.69)
# Flex workers keep billing through the idle window after a request returns,
# so it belongs in any honest per-clip figure.
IDLE_TIMEOUT_SEC = _env_float("IDLE_TIMEOUT_SEC", 5.0)

MAX_AUDIO_SECONDS = _env_float("MAX_AUDIO_SECONDS", 60.0)
MAX_DOWNLOAD_MB = _env_float("MAX_DOWNLOAD_MB", 200.0)
DOWNLOAD_TIMEOUT_SEC = _env_float("DOWNLOAD_TIMEOUT_SEC", 120.0)
FFMPEG_TIMEOUT_SEC = _env_float("FFMPEG_TIMEOUT_SEC", 300.0)
WORK_DIR = pathlib.Path(os.environ.get("WORK_DIR", "/tmp/latentsync"))
EAGER_LOAD = _env_bool("EAGER_LOAD", True)

# RunPod's documented response ceiling is 10 MB on /run and 20 MB on /runsync.
# We only warn: a truncated measurement run is worse than a large payload.
BASE64_WARN_BYTES = _env_float("BASE64_WARN_MB", 8.0) * 1024**2

VIDEO_FPS = 25  # fixed by the model; config.data.video_fps

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("latentsync-handler")

# ---------------------------------------------------------------------------
# worker lifecycle state
#
# RunPod bills a flex worker from process start, so init time is a real cost
# that the first request must carry. Tracking it separately is what lets
# bench.py report cold and warm prices apart from each other.
# ---------------------------------------------------------------------------
_PROCESS_START = time.monotonic()
_pipeline = None
_pipeline_meta: dict = {}
_pipeline_lock = threading.Lock()
# Pure pipeline construction time, and the offset from process start to the
# pipeline being ready (which additionally covers the python/torch import cost).
# Both are diagnostics; the billable figure is derived from process uptime.
_build_seconds: float | None = None
_ready_offset: float | None = None
_requests_served = 0


class MediaError(RuntimeError):
    """Bad or unusable input. Distinguished from infrastructure failures."""


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def _build_pipeline():
    """Construct the LatentSync pipeline, mirroring upstream scripts/inference.py.

    Kept deliberately faithful to the pinned commit (a229c394): the VAE scaling
    override, the dtype gate and the DeepCache parameters all come from there.
    scripts/prefetch.py verifies these signatures at build time.
    """
    from accelerate.utils import set_seed  # noqa: F401  (imported for parity)
    from DeepCache import DeepCacheSDHelper
    from diffusers import AutoencoderKL, DDIMScheduler
    from omegaconf import OmegaConf

    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature

    config = OmegaConf.load(UNET_CONFIG_PATH)

    # Upstream gate: fp16 only on compute capability > 7, i.e. Ampere or newer.
    # On a T4 (7.5) or V100 (7.0) this silently becomes fp32 and 1.6 will not
    # fit in 24 GB -- so it is worth shouting about.
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32
    if not is_fp16_supported:
        log.warning(
            "fp16 unsupported on this GPU (compute capability <= 7); falling back to "
            "fp32. LatentSync 1.6 needs ~18 GB in fp16 and will likely OOM in fp32. "
            "Use an Ampere-or-newer 24 GB GPU (L4 / A5000 / RTX 3090)."
        )

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise RuntimeError(f"unsupported cross_attention_dim={config.model.cross_attention_dim}")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        CHECKPOINT_PATH,
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    if ENABLE_DEEPCACHE:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
        log.info("DeepCache enabled (cache_interval=3, cache_branch_id=0)")

    meta = {
        "dtype": str(dtype).replace("torch.", ""),
        "num_frames": int(config.data.num_frames),
        "resolution": int(config.data.resolution),
        "mask_image_path": str(config.data.mask_image_path),
        "deepcache": ENABLE_DEEPCACHE,
    }
    return pipeline, meta


def get_pipeline():
    """Return the process-wide pipeline, building it at most once."""
    global _pipeline, _pipeline_meta, _build_seconds, _ready_offset

    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                log.info("building pipeline (config=%s)", UNET_CONFIG_PATH)
                started = time.monotonic()
                _pipeline, _pipeline_meta = _build_pipeline()
                _build_seconds = time.monotonic() - started
                _ready_offset = time.monotonic() - _PROCESS_START
                log.info(
                    "pipeline built in %.1fs; ready %.1fs after process start",
                    _build_seconds,
                    _ready_offset,
                )
    return _pipeline


def _check_onnxruntime_providers() -> None:
    """Warn if InsightFace will run face detection on CPU.

    onnxruntime falls back to the CPU provider with nothing but a warning. That
    failure mode does not break the output, it just makes face detection
    dominate the runtime -- which would quietly invalidate the cost numbers.
    """
    try:
        import onnxruntime
    except ImportError:
        log.warning("onnxruntime not importable; face detection will fail")
        return

    providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        log.error(
            "=" * 72 + "\nCUDAExecutionProvider unavailable (providers=%s).\n"
            "InsightFace will run on CPU and the measured cost will be wrong.\n"
            "Check that torch's bundled NVIDIA libs are on the loader path.\n" + "=" * 72,
            providers,
        )
    else:
        log.info("onnxruntime providers: %s", providers)


# ---------------------------------------------------------------------------
# media helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise MediaError(f"{cmd[0]} failed (exit {proc.returncode}):\n" + "\n".join(tail))
    return proc


def _probe_duration(path: pathlib.Path) -> float:
    proc = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        timeout=60,
    )
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise MediaError(f"could not read duration of {path.name}: {exc}") from exc


def _download(url: str, dest: pathlib.Path) -> None:
    import requests

    limit = MAX_DOWNLOAD_MB * 1024**2
    if not url.lower().startswith(("http://", "https://")):
        raise MediaError(f"unsupported URL scheme: {url[:40]}")

    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC) as resp:
        resp.raise_for_status()
        written = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                written += len(chunk)
                if written > limit:
                    raise MediaError(f"{url[:60]} exceeds MAX_DOWNLOAD_MB={MAX_DOWNLOAD_MB}")
                fh.write(chunk)
    if dest.stat().st_size == 0:
        raise MediaError(f"downloaded 0 bytes from {url[:60]}")


def _copy_local(rel: str, dest: pathlib.Path) -> None:
    """Copy a file shipped inside the image, e.g. the repo's demo assets.

    Benchmarking against a baked-in file removes download variance from the
    timings entirely. Confined to the repo root so this stays a test affordance
    rather than an arbitrary file read.
    """
    src = (LATENTSYNC_ROOT / rel).resolve()
    if not src.is_relative_to(LATENTSYNC_ROOT.resolve()):
        raise MediaError(f"path must be inside {LATENTSYNC_ROOT}: {rel}")
    if not src.is_file():
        raise MediaError(f"no such file in image: {rel}")
    shutil.copyfile(src, dest)


def _materialize(payload: dict, keys: tuple[str, ...], dest: pathlib.Path) -> str:
    """Resolve the first present input key into a local file. Returns its kind."""
    for key in keys:
        value = payload.get(key)
        if not value:
            continue
        if key.endswith("_base64"):
            dest.write_bytes(base64.b64decode(value))
        elif key.endswith("_path"):
            _copy_local(value, dest)
        else:
            _download(value, dest)
        return key
    raise MediaError(f"one of {list(keys)} is required")


def _normalize_audio(src: pathlib.Path, dest: pathlib.Path) -> None:
    """Mono 16 kHz PCM, matching config.data.audio_sample_rate."""
    _run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(dest),
        ],
        timeout=FFMPEG_TIMEOUT_SEC,
    )


def _normalize_video(src: pathlib.Path, dest: pathlib.Path, audio_sec: float) -> None:
    """Re-encode to 25 fps with even dimensions, looping if shorter than the audio.

    Normalizing up front is cheap insurance: a run that dies on an odd frame
    size or a short clip still burns GPU seconds, and a failed paid run teaches
    us nothing about cost.
    """
    video_sec = _probe_duration(src)
    cmd = ["ffmpeg", "-y"]
    if video_sec < audio_sec - 0.05:
        log.info("video %.2fs shorter than audio %.2fs; looping", video_sec, audio_sec)
        cmd += ["-stream_loop", "-1"]
    cmd += [
        "-i", str(src),
        "-t", f"{audio_sec:.3f}",
        "-r", str(VIDEO_FPS),
        "-an",
        # Cap the long edge and force even dimensions in one pass. The commas
        # inside min() are escaped for ffmpeg's filter parser.
        "-vf", r"scale=w=trunc(min(1280\,iw)/2)*2:h=-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(dest),
    ]
    _run(cmd, timeout=FFMPEG_TIMEOUT_SEC)


def _image_to_video(src: pathlib.Path, dest: pathlib.Path, audio_sec: float) -> None:
    """Loop a still image into a 25 fps clip as long as the audio."""
    _run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", str(src),
            "-t", f"{audio_sec:.3f}",
            "-r", str(VIDEO_FPS),
            "-vf", r"scale=w=trunc(min(1280\,iw)/2)*2:h=-2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(dest),
        ],
        timeout=FFMPEG_TIMEOUT_SEC,
    )


# ---------------------------------------------------------------------------
# cost accounting
# ---------------------------------------------------------------------------
def _cost_report(billable_sec: float) -> dict:
    usd_per_sec = GPU_USD_PER_HOUR / 3600.0
    return {
        "gpu_usd_per_hour": round(GPU_USD_PER_HOUR, 4),
        "usd_per_second": round(usd_per_sec, 9),
        "billable_sec": round(billable_sec, 2),
        "usd_this_request": round(billable_sec * usd_per_sec, 6),
        "idle_timeout_sec": IDLE_TIMEOUT_SEC,
        "usd_including_idle_timeout": round(
            (billable_sec + IDLE_TIMEOUT_SEC) * usd_per_sec, 6
        ),
        "note": (
            "billable_sec includes worker init on a cold request. Flex workers "
            "also bill through the idle window after the response is returned."
        ),
    }


def _vram_report() -> dict:
    """Summarize GPU memory. Never raises: this is also called from the OOM path,
    where a secondary failure would mask the original error."""
    if not torch.cuda.is_available():
        return {"available": False}
    try:
        _free, total = torch.cuda.mem_get_info()
        return {
            "available": True,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability()),
            # reserved is the allocator's real footprint and the number that
            # decides whether a smaller GPU tier would survive.
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "total_gb": round(total / 1024**3, 2),
        }
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}


def _progress(job: dict, message: str) -> None:
    try:
        runpod.serverless.progress_update(job, message)
    except Exception:  # local runs have no job context
        pass


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------
def handler(job: dict) -> dict:
    global _requests_served

    request_start = time.monotonic()
    job_id = str(job.get("id", "local"))
    payload = job.get("input") or {}
    is_cold = _requests_served == 0

    work = WORK_DIR / f"{job_id}-{uuid.uuid4().hex[:8]}"
    timings: dict[str, float] = {}

    try:
        work.mkdir(parents=True, exist_ok=True)
        temp_dir = work / "temp"
        temp_dir.mkdir()

        _progress(job, "loading model")
        t0 = time.monotonic()
        pipeline = get_pipeline()
        timings["model_wait"] = round(time.monotonic() - t0, 2)

        # --- fetch inputs -------------------------------------------------
        _progress(job, "downloading inputs")
        t0 = time.monotonic()
        raw_audio = work / "audio_in"
        raw_visual = work / "visual_in"
        _materialize(payload, ("audio_url", "audio_base64", "audio_path"), raw_audio)
        visual_kind = _materialize(
            payload,
            (
                "video_url", "video_base64", "video_path",
                "image_url", "image_base64", "image_path",
            ),
            raw_visual,
        )
        timings["download"] = round(time.monotonic() - t0, 2)

        # --- normalize ----------------------------------------------------
        _progress(job, "preprocessing")
        t0 = time.monotonic()
        audio_path = work / "audio.wav"
        _normalize_audio(raw_audio, audio_path)
        audio_sec = _probe_duration(audio_path)

        if audio_sec > MAX_AUDIO_SECONDS:
            raise MediaError(
                f"audio is {audio_sec:.1f}s, over MAX_AUDIO_SECONDS={MAX_AUDIO_SECONDS}. "
                "Raise the limit deliberately -- cost scales with duration."
            )
        if audio_sec <= 0.1:
            raise MediaError(f"audio too short: {audio_sec:.2f}s")

        video_path = work / "video.mp4"
        if visual_kind.startswith("image"):
            _image_to_video(raw_visual, video_path, audio_sec)
        else:
            _normalize_video(raw_visual, video_path, audio_sec)
        timings["preprocess"] = round(time.monotonic() - t0, 2)

        # --- inference ----------------------------------------------------
        from accelerate.utils import set_seed

        set_seed(SEED)

        steps = int(payload.get("inference_steps", DEFAULT_INFERENCE_STEPS))
        guidance = float(payload.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
        out_path = work / "out.mp4"

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        _progress(job, f"inference ({steps} steps)")
        t0 = time.monotonic()
        pipeline(
            video_path=str(video_path),
            audio_path=str(audio_path),
            video_out_path=str(out_path),
            num_frames=_pipeline_meta["num_frames"],
            num_inference_steps=steps,
            guidance_scale=guidance,
            weight_dtype=torch.float16
            if _pipeline_meta["dtype"] == "float16"
            else torch.float32,
            width=_pipeline_meta["resolution"],
            height=_pipeline_meta["resolution"],
            mask_image_path=_pipeline_meta["mask_image_path"],
            temp_dir=str(temp_dir),
        )
        timings["inference"] = round(time.monotonic() - t0, 2)

        if not out_path.exists():
            raise RuntimeError("pipeline finished but produced no output file")
        video_bytes = out_path.stat().st_size

        # --- encode output ------------------------------------------------
        t0 = time.monotonic()
        return_video = bool(payload.get("return_video", True))
        video_b64 = (
            base64.b64encode(out_path.read_bytes()).decode("ascii") if return_video else None
        )
        timings["encode"] = round(time.monotonic() - t0, 2)

        total = time.monotonic() - request_start
        timings["total"] = round(total, 2)

        # A flex worker is billed from process start, so on the first request
        # the whole uptime is chargeable -- model load and imports included.
        # Using uptime rather than (init + total) avoids both double counting
        # and the inflation that a from-process-start init measurement would
        # suffer if the model were loaded lazily on an already-idle worker.
        uptime = time.monotonic() - _PROCESS_START
        billable = uptime if is_cold else total

        result = {
            "timings_sec": timings,
            "worker": {
                "cold_start": is_cold,
                "model_build_sec": round(_build_seconds, 2) if _build_seconds else None,
                "ready_after_sec": round(_ready_offset, 2) if _ready_offset else None,
                "requests_served": _requests_served + 1,
                "uptime_sec": round(uptime, 2),
            },
            "vram": _vram_report(),
            "cost": _cost_report(billable),
            "media": {
                "audio_sec": round(audio_sec, 2),
                "frames": int(round(audio_sec * VIDEO_FPS)),
                "fps": VIDEO_FPS,
                "output_bytes": video_bytes,
                "source": visual_kind,
                # The one number needed to project cost to any clip length.
                "realtime_factor": round(timings["inference"] / audio_sec, 2),
            },
            "settings": {
                "inference_steps": steps,
                "guidance_scale": guidance,
                "seed": SEED,
                **_pipeline_meta,
            },
        }
        if return_video:
            result["video_base64"] = video_b64
            if video_bytes * 4 / 3 > BASE64_WARN_BYTES:
                result["warning"] = (
                    f"base64 payload is ~{video_bytes * 4 / 3 / 1024**2:.1f} MB and may "
                    "exceed RunPod's response limit (10 MB /run, 20 MB /runsync). "
                    "Pass return_video=false to measure cost without the payload."
                )

        log.info(
            "job %s done in %.1fs (inference %.1fs, rtf %.2f, peak %.1f GB) -> $%.4f",
            job_id,
            total,
            timings["inference"],
            result["media"]["realtime_factor"],
            result["vram"].get("peak_reserved_gb", 0),
            result["cost"]["usd_this_request"],
        )
        return result

    except MediaError as exc:
        log.warning("job %s rejected: %s", job_id, exc)
        return {"error": str(exc), "error_type": "bad_input"}

    except torch.cuda.OutOfMemoryError as exc:
        log.error("job %s OOM: %s", job_id, exc)
        return {
            # Recycle the worker: the allocator is fragmented and later requests
            # on this worker would fail too, each one still costing money.
            "refresh_worker": True,
            "error": (
                "CUDA out of memory. LatentSync 1.6 needs ~18 GB in fp16; use a "
                "24 GB Ampere-or-newer GPU and check vram.total_gb."
            ),
            "error_type": "cuda_oom",
            "detail": str(exc)[:500],
            "vram": _vram_report(),
        }

    except Exception as exc:
        log.error("job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "error_type": "inference_error",
            "traceback": traceback.format_exc()[-2000:],
            "timings_sec": timings,
        }

    finally:
        _requests_served += 1
        # Container disk is small and shared across requests; a leak here fills
        # it up and every later request on this worker fails.
        shutil.rmtree(work, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _startup() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    # Clear anything a previously killed worker left behind.
    for leftover in WORK_DIR.iterdir():
        shutil.rmtree(leftover, ignore_errors=True)

    _check_onnxruntime_providers()

    if EAGER_LOAD:
        try:
            get_pipeline()
        except Exception:
            # Don't take the worker down at import: let the first request
            # surface the real error through the normal error path.
            log.error("eager model load failed:\n%s", traceback.format_exc())


_startup()

if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
            # One inference at a time per worker: the GPU is the bottleneck and
            # overlapping requests would both distort timings and risk OOM.
            "concurrency_modifier": lambda _current: 1,
        }
    )
