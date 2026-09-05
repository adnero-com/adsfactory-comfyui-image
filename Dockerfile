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
EXPOSE 8188
# Same flags as the local compose service, minus --extra-model-paths-config:
# the network volume mounts straight at /ComfyUI/models.
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--cache-lru", "2", "--reserve-vram", "1"]
