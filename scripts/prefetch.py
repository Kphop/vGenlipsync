#!/usr/bin/env python3
"""Build-time asset prefetch and upstream contract verification.

Two jobs, both of which exist to protect the cost measurement:

1. Bake every model asset into the image. A download that happens inside a
   request is billed as GPU seconds, so a lazily-fetched weight would inflate
   the per-clip cost we are trying to measure -- and InsightFace lazily fetches
   a ~280 MB zip from a host that is frequently unreachable.

2. Assert the upstream API still looks the way handler.py expects. LatentSync's
   pipeline __call__ ends in **kwargs, which means a renamed or removed keyword
   is silently swallowed rather than raising. Checking signatures here turns
   that class of breakage into a failed `docker build` instead of a plausible
   but wrong video produced on a paid GPU.

Run as a Docker build step. Exits non-zero on any problem.
"""

from __future__ import annotations

import glob
import inspect
import os
import pathlib
import subprocess
import sys
import typing

LATENTSYNC_ROOT = pathlib.Path(os.environ.get("LATENTSYNC_ROOT", "/opt/latentsync"))
CHECKPOINTS = LATENTSYNC_ROOT / "checkpoints"
UNET_CONFIG = LATENTSYNC_ROOT / "configs" / "unet" / "stage2_512.yaml"

LATENTSYNC_REPO = "ByteDance/LatentSync-1.6"
VAE_REPO = "stabilityai/sd-vae-ft-mse"

# InsightFace's FaceAnalysis is constructed upstream with
#   allowed_modules=["detection", "landmark_2d_106"]
# so only these two of buffalo_l's five models are ever used. Fetching just
# these keeps ~250 MB of recognition/genderage weights out of the image.
BUFFALO_FILES = ("det_10g.onnx", "2d106det.onnx")

# Tried in order. The canonical source (storage.insightface.ai) is omitted
# deliberately: it serves only the full zip and has a poor uptime record.
BUFFALO_MIRRORS = (
    ("public-data/insightface", "models/buffalo_l/{name}"),
    ("lithiumice/insightface", "models/buffalo_l/{name}"),
    ("KwaiVGI/LivePortrait", "insightface/models/buffalo_l/{name}"),
)

MIN_ONNX_BYTES = 1_000_000


def log(msg: str) -> None:
    print(f"[prefetch] {msg}", flush=True)


def fail(msg: str) -> typing.NoReturn:
    print(f"[prefetch] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# --------------------------------------------------------------------------
# step 0: make torch's bundled NVIDIA libraries visible to onnxruntime-gpu
# --------------------------------------------------------------------------
def configure_dynamic_linker() -> None:
    """Register torch's pip-installed NVIDIA .so directories with ldconfig.

    torch finds its own CUDA libraries through RPATH, but onnxruntime-gpu
    resolves cuDNN/cuBLAS through the normal loader path. Without this,
    onnxruntime drops to CPUExecutionProvider with only a warning and face
    detection becomes the dominant cost in every request.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - build-time only
        fail(f"cannot import torch: {exc}")

    site_packages = pathlib.Path(torch.__file__).resolve().parent.parent
    lib_dirs = sorted(glob.glob(str(site_packages / "nvidia" / "*" / "lib")))

    if not lib_dirs:
        log(f"WARNING: no nvidia/*/lib directories under {site_packages}")
        return

    conf = pathlib.Path("/etc/ld.so.conf.d/zz-torch-nvidia.conf")
    conf.write_text("\n".join(lib_dirs) + "\n", encoding="utf-8")
    subprocess.run(["ldconfig"], check=True)
    log(f"registered {len(lib_dirs)} NVIDIA lib dirs with ldconfig")


# --------------------------------------------------------------------------
# step 1-3: weights
# --------------------------------------------------------------------------
def whisper_filename() -> str:
    """Pick the Whisper checkpoint the config actually asks for.

    Upstream selects tiny vs small from model.cross_attention_dim. Reading it
    rather than hardcoding means a config change can't leave us baking the
    wrong encoder (which would fail at request time, not build time).
    """
    from omegaconf import OmegaConf

    if not UNET_CONFIG.exists():
        fail(f"unet config missing: {UNET_CONFIG}")

    dim = OmegaConf.load(UNET_CONFIG).model.cross_attention_dim
    if dim == 384:
        name = "whisper/tiny.pt"
    elif dim == 768:
        name = "whisper/small.pt"
    else:
        fail(f"unexpected cross_attention_dim={dim}; upstream supports 384 or 768")
    log(f"cross_attention_dim={dim} -> {name}")
    return name


def fetch_latentsync_weights() -> None:
    from huggingface_hub import hf_hub_download

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for filename in ("latentsync_unet.pt", whisper_filename()):
        log(f"downloading {LATENTSYNC_REPO}:{filename}")
        path = hf_hub_download(
            repo_id=LATENTSYNC_REPO,
            filename=filename,
            local_dir=str(CHECKPOINTS),
        )
        size_mb = os.path.getsize(path) / 1024**2
        log(f"  -> {path} ({size_mb:.1f} MB)")


def fetch_buffalo_l() -> None:
    """Fetch the two InsightFace models upstream actually loads.

    Destination is {FaceAnalysis root}/models/buffalo_l/, where the upstream
    root is the relative path "checkpoints/auxiliary".
    """
    from huggingface_hub import hf_hub_download

    dest_dir = CHECKPOINTS / "auxiliary" / "models" / "buffalo_l"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for name in BUFFALO_FILES:
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size >= MIN_ONNX_BYTES:
            log(f"{name} already present, skipping")
            continue

        errors = []
        for repo_id, template in BUFFALO_MIRRORS:
            filename = template.format(name=name)
            try:
                log(f"downloading {repo_id}:{filename}")
                src = hf_hub_download(repo_id=repo_id, filename=filename)
            except Exception as exc:
                errors.append(f"{repo_id}: {type(exc).__name__}: {exc}")
                continue

            size = os.path.getsize(src)
            # Guard against an LFS pointer or an HTML error page landing here;
            # either would make FaceAnalysis fail with a confusing onnx error.
            if size < MIN_ONNX_BYTES:
                errors.append(f"{repo_id}: implausible size {size} bytes")
                continue

            dest.write_bytes(pathlib.Path(src).read_bytes())
            log(f"  -> {dest} ({size / 1024**2:.1f} MB)")
            break
        else:
            fail(f"could not fetch {name} from any mirror:\n  " + "\n  ".join(errors))


def fetch_vae() -> None:
    """Cache the VAE that upstream loads by hub id at pipeline build time."""
    from huggingface_hub import snapshot_download

    log(f"downloading {VAE_REPO}")
    path = snapshot_download(
        repo_id=VAE_REPO,
        allow_patterns=["*.json", "*.safetensors"],
    )
    if not glob.glob(os.path.join(path, "*.safetensors")):
        log("no safetensors in VAE snapshot, retrying for the .bin weights")
        path = snapshot_download(repo_id=VAE_REPO, allow_patterns=["*.json", "*.bin"])
        if not glob.glob(os.path.join(path, "*.bin")):
            fail(f"VAE snapshot at {path} contains no weight file")
    log(f"  -> {path}")


# --------------------------------------------------------------------------
# step 4: contract verification
# --------------------------------------------------------------------------
def _named_params(func) -> set:
    return {
        name
        for name, p in inspect.signature(func).parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


def verify_contract() -> None:
    """Fail the build if handler.py's assumptions about upstream no longer hold."""
    os.chdir(LATENTSYNC_ROOT)
    # sys.path[0] is this script's directory, and cwd is never added for a
    # `python3 path/to/script.py` invocation -- so chdir alone does not make the
    # `latentsync` package importable. Do it explicitly rather than relying on
    # PYTHONPATH being exported by the caller.
    if str(LATENTSYNC_ROOT) not in sys.path:
        sys.path.insert(0, str(LATENTSYNC_ROOT))

    try:
        from diffusers import AutoencoderKL, DDIMScheduler
        from DeepCache import DeepCacheSDHelper
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature
    except Exception as exc:
        fail(f"upstream import failed: {type(exc).__name__}: {exc}")

    # DDIMScheduler.from_pretrained("configs") is a relative path upstream; this
    # both validates the config directory and proves the cwd convention works.
    try:
        DDIMScheduler.from_pretrained("configs")
    except Exception as exc:
        fail(f'DDIMScheduler.from_pretrained("configs") failed: {exc}')

    expected = {
        LipsyncPipeline.__init__: {"vae", "audio_encoder", "unet", "scheduler"},
        Audio2Feature.__init__: {"model_path", "device", "num_frames", "audio_feat_length"},
        DeepCacheSDHelper.set_params: {"cache_interval", "cache_branch_id"},
        # __call__ ends in **kwargs, so a dropped keyword would be swallowed
        # silently. These are exactly the keywords handler.py passes.
        LipsyncPipeline.__call__: {
            "video_path",
            "audio_path",
            "video_out_path",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "weight_dtype",
            "width",
            "height",
            "mask_image_path",
            "temp_dir",
        },
    }

    for func, required in expected.items():
        missing = required - _named_params(func)
        if missing:
            fail(
                f"{func.__qualname__} no longer accepts {sorted(missing)}; "
                "handler.py must be updated to match upstream"
            )
        log(f"{func.__qualname__} signature OK")

    if not callable(getattr(UNet3DConditionModel, "from_pretrained", None)):
        fail("UNet3DConditionModel.from_pretrained is missing")

    required_files = [
        CHECKPOINTS / "latentsync_unet.pt",
        CHECKPOINTS / whisper_filename(),
        LATENTSYNC_ROOT / "latentsync" / "utils" / "mask.png",
        *(CHECKPOINTS / "auxiliary" / "models" / "buffalo_l" / n for n in BUFFALO_FILES),
    ]
    for path in required_files:
        if not path.exists():
            fail(f"expected asset missing after prefetch: {path}")
    log(f"verified {len(required_files)} on-disk assets")

    # Prove the VAE really loads from the baked cache with the network cut off.
    # Upstream loads it by hub id, so without this check a missing cache file
    # would only surface as a per-request download on a paid GPU -- or as a hard
    # failure once HF_HUB_OFFLINE is set on the image.
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        AutoencoderKL.from_pretrained(VAE_REPO)
    except Exception as exc:
        fail(
            f"offline load of {VAE_REPO} failed: {type(exc).__name__}: {exc}\n"
            "The baked HF cache is incomplete, so the VAE would be fetched at "
            "request time and billed as GPU seconds."
        )
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
    log(f"{VAE_REPO} loads offline from the baked cache")

    # Build-time machines have no GPU, so this reflects how onnxruntime was
    # built rather than whether CUDA will work. handler.py re-checks at startup.
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        log(f"onnxruntime providers: {providers}")
        if "CUDAExecutionProvider" not in providers:
            fail(
                "onnxruntime has no CUDAExecutionProvider -- the CPU-only wheel "
                "was installed. Face detection would dominate runtime and the "
                "measured cost would be meaningless."
            )
    except ImportError as exc:
        fail(f"onnxruntime not importable: {exc}")


def main() -> None:
    log(f"LATENTSYNC_ROOT={LATENTSYNC_ROOT}  HF_HOME={os.environ.get('HF_HOME')}")
    configure_dynamic_linker()
    fetch_latentsync_weights()
    fetch_buffalo_l()
    fetch_vae()
    verify_contract()
    log("all assets baked and contract verified")


if __name__ == "__main__":
    main()
