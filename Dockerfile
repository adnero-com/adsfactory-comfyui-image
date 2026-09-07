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

# --- SageAttention 2.2 (near-lossless attention speedup) -----------------------------
# HOPPER (sm_90: H200 / H100) SINGLE-ARCH build. History: the first cut targeted the
# B200 (sm_100) and the kernels COMPILED (setup.py main has HAS_SM100), but the 2.2
# public API never dispatches them -- core.py's sageattn() routes sm80/86/89/90/120/121
# and raises `Unsupported CUDA architecture: sm100` at the first attention call
# (canary job 34). So sage renders run on Hopper pods; the local 5090 has its own
# separate compose and is out of scope. Built FROM SOURCE (no Linux/py312/cu130
# prebuilt wheel). Enables the WanVideoWrapper acceleration levers (attention_mode=
# sageattn / radial_sage_attention, + torch.compile via Triton). See
# docs/design/sageattention-blackwell-build.md.
#   * Pinned `main` COMMIT (== latest at research time). Version string 2.2.0 (2++,
#     recommended over Sage3 for precision-sensitive video).
#   * TORCH_CUDA_ARCH_LIST="9.0" -> sm_90 ONLY: setup.py maps 9.0 -> HAS_SM90 with
#     gencode compute_90a/sm_90a (Hopper wgmma) and builds the sm90 kernel set
#     (_qattn_sm90: pybind_sm90.cpp + qk_int_sv_f8_cuda_sm90.cu) plus the shared
#     int8/fp8 extensions. Needs CUDA >= 12.3 (we install 13.0).
#   * RUNTIME backend: the render selects sage via WanVideoWrapper's attention_mode
#     (its own attention.py calls sage's `sageattn` AUTO dispatcher -- NOT a
#     hardcoded backend, NOT the `--use-sage-attention` Triton flag; both are known
#     to abort/blacken on Wan). On sm_90 the dispatcher calls
#     sageattn_qk_int8_pv_fp8_cuda_sm90 with pv_accum_dtype="fp32+fp32" -- the most
#     accurate accumulation path in core.py, so no accuracy patch is needed. Triton
#     ships with torch's cu130 wheels (pytorch-triton) -- also what
#     WanVideoTorchCompile needs.
#   * nvcc matching torch's cu130 ABI: install just the CUDA 13.0 compiler + runtime-dev
#     + math-lib dev headers (cuda-nvcc + cuda-cudart-dev + cuda-libraries-dev; the last
#     for torch's ATen headers which #include <cusparse.h>/<cublas...>, though sage's own
#     kernels use none of cub/thrust/cublas -- this avoids the full toolkit's
#     nsight/openjdk ~4GB) + python3-dev (Python.h, missing on the runtime base).
#   * torch 2.14's ATen headers REQUIRE C++20 but sage hardcodes -std=c++17; sed it to
#     c++20 before building (ubuntu24.04 host g++ 13 supports it).
#   * MAX_JOBS=1: fp8/int8 kernels are RAM-heavy; the ubuntu-latest CI runner (~7GB)
#     OOMs with parallel nvcc. Serial compile + the workflow's swapfile step keeps it in
#     memory (single-arch already halves the load). Raise on a big-RAM builder.
#   * LIBRARY_PATH=${CUDA_HOME}/lib64/stubs: _qattn_sm90 is the one extension that
#     links -lcuda (the CUDA DRIVER library -- its TMA/CUtensorMap path uses the
#     driver API); a GPU-less builder has no driver, so link against the stub that
#     cuda-driver-dev ships (same trick NVIDIA's own -devel images use). At runtime
#     the pod's NVIDIA container runtime injects the real libcuda.so.
# Sage correctness on Hopper still needs eyeballing: wrapper issue #1554 reported
# pure-noise output on an H100 with WanVideo (though with older builds/backends; our
# dispatcher path is the fp32+fp32 sm90 kernel) -> validate with a render whose
# OUTPUT IS VISUALLY CORRECT before trusting it; SDPA is the default fallback.
ARG SAGEATTENTION_COMMIT=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
ENV CUDA_HOME=/usr/local/cuda-13.0
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-libraries-dev-13-0 python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/thu-ml/SageAttention /tmp/SageAttention \
    && git -C /tmp/SageAttention checkout ${SAGEATTENTION_COMMIT} \
    && sed -i 's/c++17/c++20/g' /tmp/SageAttention/setup.py \
    && TORCH_CUDA_ARCH_LIST="9.0" EXT_PARALLEL=1 NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=1 \
       PATH=${CUDA_HOME}/bin:${PATH} \
       LIBRARY_PATH=${CUDA_HOME}/lib64/stubs \
       python3 -m pip install /tmp/SageAttention --no-build-isolation \
    && rm -rf /tmp/SageAttention
RUN python3 -c "import triton, sageattention; from sageattention import sageattn; from importlib.metadata import version; print('sageattention', version('sageattention'), 'triton', triton.__version__)"

# The custom_nodes tree itself (bind-mounted locally; baked here for RunPod).
ADD custom_nodes.tar.gz /ComfyUI/

WORKDIR /ComfyUI
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

# --- entrypoint (LAST on purpose) ------------------------------------------------
# Batch-parallel rendering (adsfactory Phase 7): COMFY_INSTANCES (pod env,
# default 1 -- unchanged single-process behaviour) spawns that many independent
# ComfyUI processes on ports 8188.. , EACH PINNED TO ITS OWN GPU
# (CUDA_VISIBLE_DEVICES=<i-th visible device>) and each with its own
# input/output/user/temp dir (models stay shared, read from the network volume)
# so adsfactory's providers/gpu_slots.py can route N concurrent renders to N
# different instances with zero filename-collision or VRAM-overcommit risk. One
# ComfyUI holding Wan/InfiniteTalk BF16 needs ~47GB of an 80GB H100, so
# COMFY_INSTANCES is clamped DOWN to the visible GPU count -- loudly, never
# silently. See entrypoint.sh (behaviour) and test_entrypoint.sh (the mapping).
#
# Kept as the FINAL layers on purpose: an entrypoint-only change then rebuilds
# three tiny layers instead of the whole ~20GB tail behind it.
EXPOSE 8188-8195
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
