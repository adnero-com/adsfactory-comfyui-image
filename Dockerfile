# AdsFactory ComfyUI image for RunPod (mirror of v1/comfyui/Dockerfile +
# Dockerfile.runpod, built in CI because local GitHub tokens lack write:packages).
#
# Contains ONLY public content: CUDA base, ComfyUI (pinned to the exact commit the
# local dev image was built from), public pip packages, and the custom_nodes tree
# (public repos + one local Transformers-5 compat patch in
# ComfyUI-WanVideoWrapper/multitalk/wav2vec2.py, shipped via custom_nodes.tar.gz).
# NO models, NO credentials: models come from the RunPod network volume mounted at
# /ComfyUI/models (the provider row's volume_mount_path).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git ffmpeg libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the commit the local adsfactory-comfyui:dev image runs (2026-09-04).
ARG COMFYUI_COMMIT=e80c1570b6b44a2557d5d8e341e05782d18c9bbb
RUN git clone https://github.com/comfyanonymous/ComfyUI /ComfyUI && \
    git -C /ComfyUI checkout ${COMFYUI_COMMIT}

# PyTorch with CUDA 13.0 (Blackwell sm_120 + comfy_kitchen optimized kernels)
RUN python3 -m pip install --upgrade --ignore-installed pip && \
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

RUN python3 -m pip install -r /ComfyUI/requirements.txt

# Custom-node Python deps (same layers as the local Dockerfile).
RUN python3 -m pip install opencv-python imageio-ffmpeg einops kornia scikit-build tqdm
RUN python3 -m pip install ftfy "accelerate>=1.2.1" "diffusers>=0.33.0" "peft>=0.17.0" \
        "sentencepiece>=0.2.0" protobuf pyloudnorm "gguf>=0.17.1" scipy GitPython toml

# The custom_nodes tree itself (bind-mounted locally; baked here for RunPod).
ADD custom_nodes.tar.gz /ComfyUI/

WORKDIR /ComfyUI
# Batch-parallel rendering (adsfactory Phase 7): COMFY_INSTANCES (pod env,
# default 1 -- unchanged single-process behaviour) spawns that many independent
# ComfyUI processes on ports 8188.. , each with its own input/output/user/temp
# dir (models stay shared, read from the network volume) so adsfactory's
# providers/gpu_slots.py can route N concurrent renders to N different
# instances with zero filename-collision risk. See entrypoint.sh.
EXPOSE 8188-8195
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# Same flags as the pre-existing single-instance CMD, now the per-instance
# default inside entrypoint.sh (COMFY_EXTRA_ARGS overrides it).
CMD ["/entrypoint.sh"]

# --- Solo avatar bake-off node packs (mirror of the local Dockerfile's bake-off
# section, adapted for a fully-baked image: the local dev image bind-mounts the
# node FILES and bakes only the deps; here BOTH are baked). Cloned at the exact
# commits the local working trees run, then the three local fixes are applied:
#   patches/stableavatar_local.patch — apply_scale_to_latent device/dtype align
#     (job 1014) + torchaudio>=2.9 save-to-path (torchcodec can't encode BytesIO)
#   patches/echomimic_local.patch    — same torchaudio save fix
#   overlay/.../echomimic_v3/src/dist/ — the xfuser shim (dist/ is gitignored
#     upstream, so a clone lacks it and the V3 import chain dies without it)
ARG STABLEAVATAR_COMMIT=65fdb640969999c3187dcaa17b3ec819a1398596
ARG ECHOMIMIC_COMMIT=3a36b00f405e0578db1f2c048b3ad0e41ff091bd
RUN git clone https://github.com/smthemex/ComfyUI_StableAvatar /ComfyUI/custom_nodes/ComfyUI_StableAvatar && \
    git -C /ComfyUI/custom_nodes/ComfyUI_StableAvatar checkout ${STABLEAVATAR_COMMIT} && \
    git clone https://github.com/smthemex/ComfyUI_EchoMimic /ComfyUI/custom_nodes/ComfyUI_EchoMimic && \
    git -C /ComfyUI/custom_nodes/ComfyUI_EchoMimic checkout ${ECHOMIMIC_COMMIT}
COPY patches/stableavatar_local.patch patches/echomimic_local.patch /tmp/
COPY overlay/ComfyUI_EchoMimic/echomimic_v3/src/dist/ /ComfyUI/custom_nodes/ComfyUI_EchoMimic/echomimic_v3/src/dist/
RUN git -C /ComfyUI/custom_nodes/ComfyUI_StableAvatar apply /tmp/stableavatar_local.patch && \
    git -C /ComfyUI/custom_nodes/ComfyUI_EchoMimic apply /tmp/echomimic_local.patch && \
    rm /tmp/stableavatar_local.patch /tmp/echomimic_local.patch

# Bake-off pip deps — same layers/pins/rationale as the local Dockerfile (see it
# for the full comments; the load-bearing pins:)
#   mediapipe==0.10.21 — last release with the legacy `mediapipe.solutions` API
#     EchoMimic imports at module load (1.0.1 and late 0.10.3x both break it)
#   eva-decord — decord fork with cp312 wheels (v3 flash path module-load import)
RUN python3 -m pip install omegaconf librosa moviepy easydict timm tomesd \
        torchdiffeq torchsde albumentations beautifulsoup4 \
        ffmpeg-python "mediapipe==0.10.21" ultralytics av lpips torchmetrics torchtyping IPython \
        eva-decord mmgp

# retina-face (EchoMimic V3 ip-mask face detection): tf.keras codebase on py3.12 ->
# tensorflow-cpu + tf-keras + TF_USE_LEGACY_KERAS=1; --no-deps so pip never drags
# GPU tensorflow; weights pre-baked (no runtime download on a fresh pod).
RUN python3 -m pip install tensorflow-cpu tf-keras gdown && \
    python3 -m pip install --no-deps retina-face
ENV TF_USE_LEGACY_KERAS=1
RUN mkdir -p /root/.deepface/weights && \
    wget -q -O /root/.deepface/weights/retinaface.h5 \
      https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5

# torchaudio >= 2.11 routes save() through torchcodec (lazy import) — 0.16.0
# supports torch >= 2.11 and pulls no deps of its own (local Dockerfile, jobs 900/901).
RUN python3 -m pip install torchcodec==0.16.0

# Import-chain stragglers traced from both packs' module-level imports (local Dockerfile).
RUN python3 -m pip install loguru matplotlib scikit-image onnxruntime
