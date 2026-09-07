#!/usr/bin/env bash
# AdsFactory batch-parallel rendering: spawn COMFY_INSTANCES (default 1 -- the
# pre-existing single-process behaviour, byte-identical) independent ComfyUI
# processes, EACH PINNED TO ITS OWN GPU (instance i -> CUDA_VISIBLE_DEVICES=<i-th
# VERIFIED device>, port COMFY_BASE_PORT+i), each with its OWN input/output/user
# directory so N concurrent renders (adsfactory's providers/gpu_slots.py picks
# which instance each job's ComfyUI HTTP calls target) can never collide on a
# filename regardless of what any two jobs happen to name their inputs/outputs.
#
# ONE INSTANCE PER GPU, ALWAYS. A single ComfyUI holding Wan/InfiniteTalk BF16
# needs ~47GB of an 80GB H100, so two instances on ONE card OOM (measured: 3
# instances on one H100 died; that setting was a B200-180GB leftover). Therefore
# COMFY_INSTANCES is CLAMPED to the number of GPUs actually usable by this
# container and the clamp is logged loudly -- overcommitting a card is a
# guaranteed OOM, so it is never done silently. Models are NOT per-instance:
# they live on the RunPod network volume mounted at /ComfyUI/models -- every
# instance reads that ONE copy off disk and loads it into its OWN card's VRAM.
#
# EVERY DEVICE TOKEN IS VERIFIED AGAINST CUDA BEFORE IT IS USED, and a pin that
# cannot be verified is never handed to ComfyUI. This is not paranoia, it is the
# 2026-09-07 incident (pod bet287qx74j8v5, 2 x H100): the first cut of the
# pinning change took a device list on FAITH (an inherited CUDA_VISIBLE_DEVICES,
# else nvidia-smi's index list) and exported one token per child. Both ComfyUI
# processes printed their startup banner and then died inside `import execution`
# (comfy/model_management.py -> torch.cuda.current_device() ->
# "RuntimeError: No CUDA GPUs are available") because the token did not name a
# device CUDA could open in THAT container. `wait -n` then propagated the crash,
# the container exited, and the RunPod proxy served 404 on 8188/8189 for the
# whole 900s boot window -- a pod that looked healthy in the log right up to the
# moment it wasn't. A wrong pin must therefore cost parallelism, NEVER
# reachability: unverifiable tokens are dropped, and if NOTHING verifies we fall
# back to exactly ONE UNPINNED instance -- byte-identical to the single-process
# behaviour that has rendered every ad to date.
#
# adsfactory talks to whichever instance it was assigned purely over HTTP
# (POST /prompt, /upload/image, GET /view, /system_stats) -- it never assumes
# a literal host path, so per-instance directories are fully transparent to it.
#
# Env:
#   COMFY_INSTANCES    how many ComfyUI processes to spawn (default 1). Clamped
#                      DOWN to the verified GPU count -- never up.
#   COMFY_BASE_PORT    port of instance 0 (default 8188); instance i -> base+i.
#   COMFY_GPU_DEVICES  explicit comma-separated CUDA_VISIBLE_DEVICES tokens to
#                      hand out, one per instance (e.g. "0,1" or two GPU-UUIDs).
#                      Escape hatch/override; normally auto-detected. STILL
#                      VERIFIED -- an override cannot smuggle in a dead pin.
#   COMFY_EXTRA_ARGS   per-instance ComfyUI flags (default the historical
#                      "--cache-lru 2 --reserve-vram 1"). --listen/--port are
#                      REFUSED here: the bind address and the per-instance port
#                      are the two things that make the pod reachable at all.
#   COMFY_CUDA_PROBE   the shell command used to verify ONE device token (run
#                      with CUDA_VISIBLE_DEVICES=<token>; exit 0 == usable).
#                      Overridden by test_entrypoint_spawn.sh; never in prod.
#
# Sourcing this file with COMFY_ENTRYPOINT_SOURCE_ONLY=1 defines the pure
# helpers below and returns WITHOUT starting anything -- that is how
# test_entrypoint.sh exercises the argv/instance/port/GPU mapping (incl. the
# clamp and the verification fallback) with no GPU, no ComfyUI and no container.
set -euo pipefail

# The bind address is a CONSTANT, not a knob. ComfyUI defaults to 127.0.0.1,
# which is unreachable from the RunPod proxy (and from anything outside the
# container) -- every instance we start must listen on every interface.
COMFY_LISTEN_ADDR="0.0.0.0"

# Verify ONE device token by asking CUDA itself, in a throwaway process, with
# exactly the CUDA_VISIBLE_DEVICES value the ComfyUI child would get. This is
# the same question ComfyUI's own `import execution` asks ~20s into its boot --
# asked here, where a "no" costs a warning instead of the whole pod.
COMFY_CUDA_PROBE_DEFAULT='python3 -c "import torch, sys; sys.exit(0 if torch.cuda.device_count() >= 1 else 1)"'

# ------------------------------------------------------------- pure helpers --

# comfy_split_csv <csv> -- one whitespace-stripped, non-empty token per line.
# The ONE place a device list is parsed (comfy_plan and comfy_verify_devices
# both go through it, so they can never disagree about what a token is).
comfy_split_csv() {
    local csv="${1:-}" tok oldifs="${IFS}"
    IFS=','
    for tok in ${csv}; do
        tok="${tok//[[:space:]]/}"
        [ -n "${tok}" ] && printf '%s\n' "${tok}"
    done
    IFS="${oldifs}"
    return 0
}

# Echo the comma-separated list of CANDIDATE device tokens this container may
# use, in the order instances should claim them. Candidates only -- every one of
# them is then put to comfy_verify_devices before it can pin anything.
# Precedence:
#   1. COMFY_GPU_DEVICES  -- explicit admin override.
#   2. CUDA_VISIBLE_DEVICES already in the container env -- the set the runtime
#      restricted us to.
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
        # so the per-GPU newlines survive for paste to join on.
        # `|| true`: a driver hiccup must degrade to "unknown GPU list" (handled
        # loudly by comfy_plan), never kill the container under `set -e`.
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
            | tr -d ' ' | paste -sd, - || true
        return 0
    fi
    printf '\n'
}

# comfy_verify_devices <candidates_csv>
#
# Echo the comma-separated SUBSET of candidates that CUDA can actually open in
# THIS container, preserving order. Each token is probed in its own process with
# CUDA_VISIBLE_DEVICES set to exactly that token -- the same environment the
# ComfyUI child would inherit -- so a token that survives here cannot fail at
# `import execution` for a reason we could have seen. Dropped tokens are logged
# loudly on stderr.
#
# A token can look perfectly reasonable and still be dead: an inherited
# CUDA_VISIBLE_DEVICES computed OUTSIDE the container indexes the HOST's device
# list, an nvidia-smi index can name a card the driver will not let this
# container open, and an admin override is just a string someone typed. Faith in
# any of those cost us pod bet287qx74j8v5. The probe costs ~5s per card, once,
# on a pod that already spent ~12 minutes pulling its image.
comfy_verify_devices() {
    local candidates="${1:-}"
    local probe="${COMFY_CUDA_PROBE:-${COMFY_CUDA_PROBE_DEFAULT}}"
    local -a ok=()
    local tok
    while IFS= read -r tok; do
        [ -n "${tok}" ] || continue
        if CUDA_VISIBLE_DEVICES="${tok}" bash -c "${probe}" >/dev/null 2>&1; then
            ok+=("${tok}")
        else
            echo "entrypoint: WARNING device token '${tok}' is NOT usable by CUDA in this container -- dropping it (a ComfyUI pinned to it would die in 'import execution' and take the pod's ports with it)" >&2
        fi
    done < <(comfy_split_csv "${candidates}")
    ( IFS=','; printf '%s\n' "${ok[*]:-}" )
}

# comfy_extra_args <raw> -- the sanitized per-instance ComfyUI flags, one token
# per line. --listen and --port are REFUSED (loudly): the bind address and the
# per-instance port are exactly what makes an instance reachable, and ComfyUI's
# argparse lets a later flag win -- a stray "--listen 127.0.0.1" in
# COMFY_EXTRA_ARGS would silently produce a process that is up, healthy in the
# log, and invisible to the proxy. Everything else passes through untouched.
comfy_extra_args() {
    local raw="${1-}" tok skip=0
    for tok in ${raw}; do
        if [ "${skip}" = "1" ]; then skip=0; continue; fi
        case "${tok}" in
            --listen|--port)
                echo "entrypoint: WARNING refusing '${tok}' from COMFY_EXTRA_ARGS -- the bind address (${COMFY_LISTEN_ADDR}) and the per-instance port are set by the entrypoint" >&2
                skip=1
                continue ;;
            --listen=*|--port=*)
                echo "entrypoint: WARNING refusing '${tok}' from COMFY_EXTRA_ARGS -- the bind address (${COMFY_LISTEN_ADDR}) and the per-instance port are set by the entrypoint" >&2
                continue ;;
        esac
        printf '%s\n' "${tok}"
    done
    return 0
}

# comfy_argv <index> <port> [extra_args]
#
# The COMPLETE argv of one ComfyUI instance, one token per line -- the single
# source of truth for what is executed, so a test can assert the real thing
# rather than a paraphrase of it. The first eleven tokens are frozen: they are
# byte-for-byte the argv the production pods have always run (captured live from
# a healthy pod's /system_stats "argv"), with only the port and the instance
# directory index varying per instance.
comfy_argv() {
    local idx="${1:-0}" port="${2:-8188}" extra="${3-}"
    local inst_dir="/ComfyUI/instances/${idx}"
    printf '%s\n' \
        python3 main.py \
        --listen "${COMFY_LISTEN_ADDR}" \
        --port "${port}" \
        --input-directory "${inst_dir}/input" \
        --output-directory "${inst_dir}/output" \
        --user-directory "${inst_dir}/user" \
        --temp-directory "${inst_dir}/temp"
    comfy_extra_args "${extra}"
}

# comfy_plan <instances> <base_port> <verified_devices_csv>
#
# The whole mapping, as ONE pure function (no side effects beyond warnings on
# stderr) so it is testable without a GPU: echoes one "<index> <port> <device>"
# line per instance to spawn. <device> is the value to export as
# CUDA_VISIBLE_DEVICES, or "-" meaning "leave the inherited environment alone"
# (the legacy single-instance behaviour when no device could be VERIFIED).
#
# Rules:
#   * a non-numeric/zero/negative instance count falls back to 1;
#   * instances are CLAMPED DOWN to the number of verified devices -- loudly.
#     Two ComfyUI processes on one card OOM, so serving fewer instances than
#     asked is always better than serving all of them broken;
#   * with NO verified device at all, exactly ONE UNPINNED instance runs: we
#     cannot prove two processes would land on different cards, and we must not
#     pin to a token CUDA rejected, so we degrade to the legacy behaviour that
#     is known to boot and bind.
comfy_plan() {
    local want="${1:-1}" base="${2:-8188}" devices_csv="${3:-}"
    local -a devices=()
    local tok
    while IFS= read -r tok; do
        [ -n "${tok}" ] && devices+=("${tok}")
    done < <(comfy_split_csv "${devices_csv}")

    if ! [[ "${want}" =~ ^[0-9]+$ ]] || [ "${want}" -lt 1 ]; then
        echo "entrypoint: COMFY_INSTANCES=${want} is not a positive integer -- defaulting to 1" >&2
        want=1
    fi

    local gpus=${#devices[@]} i
    if [ "${gpus}" -eq 0 ]; then
        echo "entrypoint: WARNING no CUDA-usable GPU could be verified (no COMFY_GPU_DEVICES, no CUDA_VISIBLE_DEVICES, no nvidia-smi, or every candidate token failed its CUDA probe)" >&2
        if [ "${want}" -gt 1 ]; then
            echo "entrypoint: WARNING refusing to start ${want} instances without a verified GPU list -- CLAMPING to 1 (two ComfyUI processes on one card OOM)" >&2
            want=1
        fi
        echo "entrypoint: starting 1 UNPINNED instance (the legacy single-process behaviour) -- ComfyUI will pick its own device" >&2
        for ((i = 0; i < want; i++)); do
            printf '%s %s %s\n' "${i}" "$((base + i))" "-"
        done
        return 0
    fi

    if [ "${want}" -gt "${gpus}" ]; then
        echo "entrypoint: WARNING COMFY_INSTANCES=${want} exceeds the ${gpus} verified GPU(s) [${devices[*]}] -- CLAMPING to ${gpus} (one ComfyUI per GPU; two on one card OOM)" >&2
        want="${gpus}"
    elif [ "${want}" -lt "${gpus}" ]; then
        echo "entrypoint: note COMFY_INSTANCES=${want} is fewer than the ${gpus} verified GPU(s) -- $((gpus - want)) GPU(s) will sit idle" >&2
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

CANDIDATES="$(comfy_gpu_devices)"
echo "entrypoint: candidate GPU device token(s): [${CANDIDATES:-none}]"
DEVICES_CSV="$(comfy_verify_devices "${CANDIDATES}")"
echo "entrypoint: CUDA-verified GPU device token(s): [${DEVICES_CSV:-none}]"

mapfile -t PLAN < <(comfy_plan "${COMFY_INSTANCES:-1}" "${BASE_PORT}" "${DEVICES_CSV}")
echo "entrypoint: starting ${#PLAN[@]} ComfyUI instance(s)"

pids=()
for line in "${PLAN[@]}"; do
    read -r idx port dev <<< "${line}"
    inst_dir="/ComfyUI/instances/${idx}"
    mkdir -p "${inst_dir}/input" "${inst_dir}/output" "${inst_dir}/user" "${inst_dir}/temp"
    mapfile -t cmd < <(comfy_argv "${idx}" "${port}" "${EXTRA_ARGS}")
    echo "entrypoint: instance ${idx} argv: ${cmd[*]}"
    if [ "${dev}" = "-" ]; then
        "${cmd[@]}" &
    else
        # Per-process pin: instance i owns exactly one VERIFIED card. Exported
        # for THIS child only (a prefix assignment), so the parent env stays
        # untouched.
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
