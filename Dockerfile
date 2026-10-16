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

# Pinned like everything else: pip's own resolver changes between releases, so
# an unpinned "--upgrade pip" is itself a source of build drift. These are the
# versions present in the image that was verified working.
RUN python3 -m pip install --upgrade pip==26.2.1 setuptools==84.0.0 wheel==0.48.0

# Install torch explicitly from the cu121 index BEFORE the repo requirements.
# The repo's requirements.txt uses --extra-index-url, which leaves pip free to
# resolve torch 2.5.1 from PyPI (a cu124 build) instead. Pinning it here keeps
# the CUDA minor version aligned with the base image.
# Every pip step below passes -c with this lock. Without it the image is only
# reproducible by accident: a rebuild on 2026-10-16 from byte-identical source
# produced workers that died before serving a job, because unpinned transitive
# dependencies had drifted over a week. Constraints cap versions for whatever
# gets installed without forcing extra installs.
COPY requirements.lock.txt ${APP_ROOT}/requirements.lock.txt

RUN python3 -m pip install -c "${APP_ROOT}/requirements.lock.txt" \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/bytedance/LatentSync.git "${LATENTSYNC_ROOT}" \
    && git -C "${LATENTSYNC_ROOT}" checkout --quiet "${LATENTSYNC_COMMIT}" \
    && rm -rf "${LATENTSYNC_ROOT}/.git"

# insightface==0.7.3 ships as an sdist and its setup.py needs Cython + numpy
# importable at build time. Installing them first avoids a build failure that
# otherwise surfaces 10 minutes into the layer.
RUN python3 -m pip install -c "${APP_ROOT}/requirements.lock.txt" "cython<3.0" "numpy==1.26.4"

# mediapipe and gradio are dropped deliberately. Neither is on the inference
# path: mediapipe is imported only by preprocess/ and eval/ (training-data and
# evaluation scripts) and gradio only by gradio_app.py, the interactive demo.
#
# They also have to go for the build to be pinnable at all. `pip check` on the
# image that ran our benchmarks reports exactly two broken requirements:
#
#   gradio 5.24.0    requires tomlkit<0.14.0, but has tomlkit 0.15.1
#   mediapipe 0.10.11 requires protobuf<4,   but has protobuf 7.36.2
#
# That image "worked" only because pip installs sequentially: these two went in
# first, and later steps upgraded their dependencies behind their backs without
# re-checking. mediapipe was in fact already broken at runtime, raising
# MessageFactory.GetPrototype on import. No single resolution can reproduce
# that state, so the set cannot be locked while they are present.
#
# Dropping them also sheds jaxlib, pandas and friends -- several hundred MB
# that nothing here loads.
RUN grep -vE '^(mediapipe|gradio)' "${LATENTSYNC_ROOT}/requirements.txt" > /tmp/reqs-trimmed.txt \
 && python3 -m pip install -c "${APP_ROOT}/requirements.lock.txt" -r /tmp/reqs-trimmed.txt \
 && rm -f /tmp/reqs-trimmed.txt

COPY requirements.txt ${APP_ROOT}/requirements.txt
RUN python3 -m pip install -c "${APP_ROOT}/requirements.lock.txt" -r "${APP_ROOT}/requirements.txt"

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
#
# Both land in the repo root, which is also WORKDIR. That matters for three
# reasons: the runpod SDK looks for test_input.json in the working directory;
# putting handler.py at the workdir root makes `latentsync` importable via
# sys.path[0] without relying on PYTHONPATH; and it lets CMD reference
# handler.py at the same relative path it occupies in this git repo, which is
# what RunPod's GitHub integration scans for when it looks for the worker.
COPY handler.py ${LATENTSYNC_ROOT}/handler.py
COPY test_input.json ${LATENTSYNC_ROOT}/test_input.json

# WORKDIR must stay at the repo root: upstream resolves several paths relatively
# (DDIMScheduler.from_pretrained("configs"), FaceAnalysis(root="checkpoints/auxiliary"),
# and config.data.mask_image_path). Running from anywhere else fails at request time.
CMD ["python3", "-u", "handler.py"]
