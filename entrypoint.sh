#!/usr/bin/env bash
# AdsFactory batch-parallel rendering: spawn COMFY_INSTANCES (default 1 -- the
# pre-existing single-process behaviour, byte-identical) independent ComfyUI
# processes, EACH PINNED TO ITS OWN GPU (instance i -> CUDA_VISIBLE_DEVICES=<i-th
# visible device>, port COMFY_BASE_PORT+i), each with its OWN input/output/user
# directory so N concurrent renders (adsfactory's providers/gpu_slots.py picks
# which instance each job's ComfyUI HTTP calls target) can never collide on a
# filename regardless of what any two jobs happen to name their inputs/outputs.
#
# ONE INSTANCE PER GPU, ALWAYS. A single ComfyUI holding Wan/InfiniteTalk BF16
# needs ~47GB of an 80GB H100, so two instances on ONE card OOM (measured: 3
# instances on one H100 died; that setting was a B200-180GB leftover). Therefore
# COMFY_INSTANCES is CLAMPED to the number of GPUs actually visible to this
# container and the clamp is logged loudly -- overcommitting a card is a
# guaranteed OOM, so it is never done silently. Models are NOT per-instance:
# they live on the RunPod network volume mounted at /ComfyUI/models -- every
# instance reads that ONE copy off disk and loads it into its OWN card's VRAM.
#
# adsfactory talks to whichever instance it was assigned purely over HTTP
# (POST /prompt, /upload/image, GET /view, /system_stats) -- it never assumes
# a literal host path, so per-instance directories are fully transparent to it.
#
# Env:
#   COMFY_INSTANCES    how many ComfyUI processes to spawn (default 1). Clamped
#                      DOWN to the visible GPU count -- never up.
#   COMFY_BASE_PORT    port of instance 0 (default 8188); instance i -> base+i.
#   COMFY_GPU_DEVICES  explicit comma-separated CUDA_VISIBLE_DEVICES tokens to
#                      hand out, one per instance (e.g. "0,1" or two GPU-UUIDs).
#                      Escape hatch/override; normally auto-detected.
#   COMFY_EXTRA_ARGS   per-instance ComfyUI flags (default the historical
#                      "--cache-lru 2 --reserve-vram 1").
#
# Sourcing this file with COMFY_ENTRYPOINT_SOURCE_ONLY=1 defines the pure
# helpers below and returns WITHOUT starting anything -- that is how
# test_entrypoint.sh exercises the instance->port->GPU mapping (incl. the clamp)
# with no GPU, no ComfyUI and no container.
set -euo pipefail

# ------------------------------------------------------------- pure helpers --

# Echo the comma-separated list of device tokens this container may use, in the
# order instances should claim them. Precedence:
#   1. COMFY_GPU_DEVICES  -- explicit admin override, always wins.
#   2. CUDA_VISIBLE_DEVICES already in the container env -- the set the runtime
#      restricted us to. We hand out ITS entries (never raw 0..N-1 indices),
#      because a per-instance CUDA_VISIBLE_DEVICES=<n> would otherwise index the
#      DRIVER's device list and could point at a card we were not given.
#   3. nvidia-smi's own index list -- the normal RunPod case (the container sees
#      exactly its rented GPUs as 0..N-1).
#   4. nothing -- no GPU visible at all (a CPU-only builder/test container).
# "all"/"none"/"void" are NVIDIA_VISIBLE_DEVICES keywords, not real CUDA device
# lists; if one leaks into CUDA_VISIBLE_DEVICES, ignore it and ask the driver.
comfy_gpu_devices() {
    if [ -n "${COMFY_GPU_DEVICES:-}" ]; then
        printf '%s\n' "${COMFY_GPU_DEVICES}"
        return 0
    fi
    local cvd="${CUDA_VISIBLE_DEVICES:-}"
    # Pure-bash lowercase (no `tr`): this helper must still work in a stripped
    # environment -- the test harness exercises exactly that.
    case "${cvd,,}" in
        ""|all|none|void) cvd="" ;;
    esac
    if [ -n "${cvd}" ]; then
        printf '%s\n' "${cvd}"
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        # One index per line -> "0,1". `tr -d ' '` (spaces only, NOT [:space:])
        # so the per-GPU newlines survive for paste to join on. CUDA's default
        # device order (FASTEST_FIRST) can differ from nvidia-smi's (PCI order),
        # but we only ever need DISTINCT cards, one per instance -- a permutation
        # of the same set still gives every instance its own GPU.
        # `|| true`: a driver hiccup must degrade to "unknown GPU list" (handled
        # loudly by comfy_plan), never kill the container under `set -e`.
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
            | tr -d ' ' | paste -sd, - || true
        return 0
    fi
    printf '\n'
}

# comfy_plan <instances> <base_port> <devices_csv>
#
# The whole mapping, as ONE pure function (no side effects beyond warnings on
# stderr) so it is testable without a GPU: echoes one "<index> <port> <device>"
# line per instance to spawn. <device> is the value to export as
# CUDA_VISIBLE_DEVICES, or "-" meaning "leave the inherited environment alone"
# (the legacy single-instance behaviour when no device list could be determined).
#
# Rules:
#   * a non-numeric/zero/negative instance count falls back to 1;
#   * instances are CLAMPED DOWN to the number of devices -- loudly. Two ComfyUI
#     processes on one card OOM, so serving fewer instances than asked is always
#     better than serving all of them broken;
#   * with NO device list at all, at most ONE instance runs (unpinned): we
#     cannot prove two processes would land on different cards, so we refuse to
#     gamble a guaranteed OOM.
comfy_plan() {
    local want="${1:-1}" base="${2:-8188}" devices_csv="${3:-}"
    local -a devices=()
    local tok oldifs="${IFS}"
    IFS=','
    for tok in ${devices_csv}; do
        tok="${tok//[[:space:]]/}"
        [ -n "${tok}" ] && devices+=("${tok}")
    done
    IFS="${oldifs}"

    if ! [[ "${want}" =~ ^[0-9]+$ ]] || [ "${want}" -lt 1 ]; then
        echo "entrypoint: COMFY_INSTANCES=${want} is not a positive integer -- defaulting to 1" >&2
        want=1
    fi

    local gpus=${#devices[@]} i
    if [ "${gpus}" -eq 0 ]; then
        echo "entrypoint: WARNING no GPU could be detected (no COMFY_GPU_DEVICES, no CUDA_VISIBLE_DEVICES, no nvidia-smi)" >&2
        if [ "${want}" -gt 1 ]; then
            echo "entrypoint: WARNING refusing to start ${want} instances without a known GPU list -- CLAMPING to 1 (two ComfyUI processes on one card OOM)" >&2
            want=1
        fi
        for ((i = 0; i < want; i++)); do
            printf '%s %s %s\n' "${i}" "$((base + i))" "-"
        done
        return 0
    fi

    if [ "${want}" -gt "${gpus}" ]; then
        echo "entrypoint: WARNING COMFY_INSTANCES=${want} exceeds the ${gpus} visible GPU(s) [${devices[*]}] -- CLAMPING to ${gpus} (one ComfyUI per GPU; two on one card OOM)" >&2
        want="${gpus}"
    elif [ "${want}" -lt "${gpus}" ]; then
        echo "entrypoint: note COMFY_INSTANCES=${want} is fewer than the ${gpus} visible GPU(s) -- $((gpus - want)) GPU(s) will sit idle" >&2
    fi

    for ((i = 0; i < want; i++)); do
        printf '%s %s %s\n' "${i}" "$((base + i))" "${devices[i]}"
    done
}

# Sourced by the test harness: stop here, before anything is started.
if [ "${COMFY_ENTRYPOINT_SOURCE_ONLY:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

# ------------------------------------------------------------------- main ----

BASE_PORT="${COMFY_BASE_PORT:-8188}"
# Same flags the single-instance CMD always ran with (Dockerfile.runpod, the
# public repo's pre-existing default) -- override via COMFY_EXTRA_ARGS if a
# future round needs different per-deployment tuning without another image change.
EXTRA_ARGS="${COMFY_EXTRA_ARGS:---cache-lru 2 --reserve-vram 1}"

DEVICES_CSV="$(comfy_gpu_devices)"
echo "entrypoint: visible GPU device list: [${DEVICES_CSV:-none}]"

mapfile -t PLAN < <(comfy_plan "${COMFY_INSTANCES:-1}" "${BASE_PORT}" "${DEVICES_CSV}")
echo "entrypoint: starting ${#PLAN[@]} ComfyUI instance(s)"

pids=()
for line in "${PLAN[@]}"; do
    read -r idx port dev <<< "${line}"
    inst_dir="/ComfyUI/instances/${idx}"
    mkdir -p "${inst_dir}/input" "${inst_dir}/output" "${inst_dir}/user" "${inst_dir}/temp"
    # shellcheck disable=SC2206  # EXTRA_ARGS is deliberately word-split
    cmd=(python3 main.py --listen 0.0.0.0 --port "${port}"
         --input-directory "${inst_dir}/input"
         --output-directory "${inst_dir}/output"
         --user-directory "${inst_dir}/user"
         --temp-directory "${inst_dir}/temp"
         ${EXTRA_ARGS})
    if [ "${dev}" = "-" ]; then
        "${cmd[@]}" &
    else
        # Per-process pin: instance i owns exactly one card. Exported for THIS
        # child only (a prefix assignment), so the parent env stays untouched.
        CUDA_VISIBLE_DEVICES="${dev}" "${cmd[@]}" &
    fi
    pid=$!
    pids+=("${pid}")
    echo "entrypoint: instance ${idx} started on port ${port} (pid ${pid}, GPU ${dev}, dirs under ${inst_dir})"
done

# If ANY instance's process exits (crash, OOM, ...), stop the rest and exit
# non-zero -- RunPod / docker's restart policy should recreate the container
# rather than silently keep serving with fewer instances than configured (a job
# routed to a dead instance's port would just hang/timeout otherwise).
set +e
wait -n "${pids[@]}"
code=$?
set -e
echo "entrypoint: an instance exited (code ${code}) -- stopping the remaining instance(s)"
kill "${pids[@]}" 2>/dev/null || true
wait || true
exit "${code}"
