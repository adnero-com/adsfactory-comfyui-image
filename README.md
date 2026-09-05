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
