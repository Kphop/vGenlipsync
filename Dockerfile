# LatentSync 1.6 on RunPod Serverless -- minimal image built for cost measurement.
#
# Base image choice: CUDA 12.1 *runtime* with no bundled cuDNN. The repo pins
# torch==2.5.1+cu121, which drags in nvidia-cudnn-cu12==9.1.0.70 via pip, and
# onnxruntime-gpu==1.21.0 needs cuDNN 9. A `-cudnn8` base would put cuDNN 8 on
# the library path and InsightFace would silently fall back to the CPU provider,
# which makes face detection crawl and corrupts the timing numbers this image
# exists to produce.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Pinned to a specific commit, not `main`. Upstream has no release tags, so a
# floating ref would make measured timings non-reproducible.
ARG LATENTSYNC_COMMIT=a229c3948406bc2cf6eaf4873e662e70c6a04746

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LATENTSYNC_ROOT=/opt/latentsync \
    APP_ROOT=/opt/app \
    HF_HOME=/opt/hf

# python3.10 is Ubuntu 22.04's system python, which is exactly what upstream
# specifies (3.10.13). Do not "upgrade" it: mediapipe==0.10.11 publishes wheels
# only up to cp311, so 3.12+ would force a source build of a pinned dep.
# libgl1 + libglib2.0-0 are for cv2; build-essential is for insightface, which
# is sdist-only on PyPI and must be compiled.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        python3.10 \
        python3.10-dev \
        python3-pip \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Install torch explicitly from the cu121 index BEFORE the repo requirements.
# The repo's requirements.txt uses --extra-index-url, which leaves pip free to
# resolve torch 2.5.1 from PyPI (a cu124 build) instead. Pinning it here keeps
# the CUDA minor version aligned with the base image.
RUN python3 -m pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/bytedance/LatentSync.git "${LATENTSYNC_ROOT}" \
    && git -C "${LATENTSYNC_ROOT}" checkout --quiet "${LATENTSYNC_COMMIT}" \
    && rm -rf "${LATENTSYNC_ROOT}/.git"

# insightface==0.7.3 ships as an sdist and its setup.py needs Cython + numpy
# importable at build time. Installing them first avoids a build failure that
# otherwise surfaces 10 minutes into the layer.
RUN python3 -m pip install "cython<3.0" "numpy==1.26.4"

RUN python3 -m pip install -r "${LATENTSYNC_ROOT}/requirements.txt"

COPY requirements.txt ${APP_ROOT}/requirements.txt
RUN python3 -m pip install -r "${APP_ROOT}/requirements.txt"

# Belt and braces with the ldconfig pass inside prefetch.py: make torch's bundled
# NVIDIA libraries discoverable so onnxruntime-gpu can find cuDNN/cuBLAS.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib

WORKDIR ${LATENTSYNC_ROOT}

# Must be set before the prefetch step, which imports `latentsync` to verify the
# upstream API contract.
ENV PYTHONPATH=${LATENTSYNC_ROOT}

# Bake every model asset into the image. Anything downloaded at request time is
# billed as GPU seconds and would contaminate the cost measurement. This step
# also verifies the upstream API contract, so an upstream refactor fails the
# build here instead of failing on a paid GPU.
COPY scripts/prefetch.py ${APP_ROOT}/scripts/prefetch.py
RUN python3 "${APP_ROOT}/scripts/prefetch.py"

# Set only after prefetch has populated and verified the cache. Keeping the hub
# offline at runtime means no cold start can be slowed by a network round-trip,
# which keeps the measured timings comparable between runs.
# Set HF_HUB_OFFLINE=0 at deploy time if you need to debug a cache miss.
ENV HF_HUB_OFFLINE=1

# Copied last so handler edits don't invalidate the pip and weight layers.
COPY handler.py ${APP_ROOT}/handler.py
# Lands in the repo root because the runpod SDK looks for test_input.json in the
# working directory, and handler.py chdir's to LATENTSYNC_ROOT.
COPY test_input.json ${LATENTSYNC_ROOT}/test_input.json

# WORKDIR must stay at the repo root: upstream resolves several paths relatively
# (DDIMScheduler.from_pretrained("configs"), FaceAnalysis(root="checkpoints/auxiliary"),
# and config.data.mask_image_path). Running from anywhere else fails at request time.
CMD ["python3", "-u", "/opt/app/handler.py"]
