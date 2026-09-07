# adsfactory-comfyui-image

Public build repo for `ghcr.io/adnero-com/adsfactory-comfyui` — the ComfyUI image
AdsFactory's RunPod render pods run (Phase 7 R1). Public so the GHCR package is
public and RunPod pods can pull it without registry credentials.

Contents are public only: CUDA base, pinned ComfyUI, public pip deps, and the
custom_nodes tree (public repos + a small Transformers-5 compat patch in
ComfyUI-WanVideoWrapper). No models, no secrets — models live on the RunPod
network volume mounted at `/ComfyUI/models`.

The source of truth for the local variant is `adsfactory/v1/comfyui/Dockerfile`
(private repo); keep the two in sync when the model stack changes.

## Concurrency: one ComfyUI per GPU

`entrypoint.sh` spawns `COMFY_INSTANCES` (default 1) ComfyUI processes: instance
`i` listens on `COMFY_BASE_PORT + i` (default 8188, 8189, ...), gets its own
`--input/--output/--user/--temp` directory, and is pinned to its own card with
`CUDA_VISIBLE_DEVICES`. Models stay shared — one copy on the network volume,
loaded independently into each card's VRAM.

`COMFY_INSTANCES` is clamped **down** to the number of GPUs the container can
see (and the clamp is logged loudly): one ComfyUI holding Wan/InfiniteTalk BF16
needs ~47GB of an 80GB H100, so two instances on one card OOM. A 2-GPU pod with
`COMFY_INSTANCES=2` therefore renders two ads concurrently; the same env on a
1-GPU pod quietly runs one instead of dying.

`./test_entrypoint.sh` covers that mapping (ports, pinning, clamp) with no GPU
and no container, and gates the image build in CI.
