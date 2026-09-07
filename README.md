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
`i` listens on `0.0.0.0:COMFY_BASE_PORT + i` (default 8188, 8189, ...), gets its
own `--input/--output/--user/--temp` directory, and is pinned to its own card
with `CUDA_VISIBLE_DEVICES`. Models stay shared — one copy on the network
volume, loaded independently into each card's VRAM.

**Every device token is verified against CUDA before it pins anything.** The
candidate list (`COMFY_GPU_DEVICES` > an inherited `CUDA_VISIBLE_DEVICES` >
`nvidia-smi`) is only a proposal: each token is probed in a throwaway process
with exactly the `CUDA_VISIBLE_DEVICES` value the ComfyUI child would get, and a
token CUDA cannot open is dropped (loudly). If nothing verifies, exactly **one
unpinned** instance runs — the legacy single-process behaviour, reachable on
8188. A wrong pin costs parallelism, never reachability.

That rule is written in blood: pod `bet287qx74j8v5` (2 × H100, 2026-09-07) took
its device list on faith, both ComfyUI processes died inside `import execution`
with `RuntimeError: No CUDA GPUs are available` seconds after logging a
perfectly healthy startup banner, `wait -n` propagated the crash, and the
RunPod proxy served 404 on 8188/8189 for the full 900s boot window.

`COMFY_INSTANCES` is clamped **down** to the number of *verified* GPUs (and the
clamp is logged loudly): one ComfyUI holding Wan/InfiniteTalk BF16 needs ~47GB
of an 80GB H100, so two instances on one card OOM. A 2-GPU pod with
`COMFY_INSTANCES=2` therefore renders two ads concurrently; the same env on a
1-GPU pod quietly runs one instead of dying.

`COMFY_EXTRA_ARGS` (default `--cache-lru 2 --reserve-vram 1`) is appended to
every instance, but `--listen` and `--port` are refused there — they are what
makes an instance reachable at all, and ComfyUI's argparse lets the last flag
win.

## Tests (both gate the image build in CI)

- `./test_entrypoint.sh` — pure helpers, no GPU/container: the **complete
  per-instance argv** frozen against a healthy production pod's own
  `GET /system_stats` `argv`, the device verification, and the
  instance → port → GPU mapping incl. the clamp.
- `sudo ./test_entrypoint_spawn.sh` — runs the **real entrypoint** against a
  stub ComfyUI that records its argv/`CUDA_VISIBLE_DEVICES` and opens a real
  socket, then asserts every instance's argv *and* that its port is connectable
  **on a non-loopback address**. This is the test that fails when the mapping is
  right but nothing is listening. Needs root (it creates `/ComfyUI` and binds
  8188/8189); in a container: `docker run --rm -v "$PWD:/w" -w /w ubuntu:24.04
  bash ./test_entrypoint_spawn.sh`.
