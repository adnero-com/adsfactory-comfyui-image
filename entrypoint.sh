#!/usr/bin/env bash
# AdsFactory batch-parallel rendering: spawn COMFY_INSTANCES (default 1 -- the
# pre-existing single-process behaviour, byte-identical) independent ComfyUI
# processes on ports COMFY_BASE_PORT..COMFY_BASE_PORT+N-1, each with its OWN
# input/output/user directory so N concurrent renders (adsfactory's
# providers/gpu_slots.py picks which instance each job's ComfyUI HTTP calls
# target) can never collide on a filename regardless of what any two jobs
# happen to name their inputs/outputs. Models are NOT per-instance: they live
# on the RunPod network volume, mounted read-only-in-practice at
# /ComfyUI/models -- every instance shares that ONE copy (that's the whole
# point: 3 x InfiniteTalk BF16 at ~46GB VRAM each fits comfortably in a B200's
# 180GB, but the ~37GB of model files on disk is loaded independently into
# each process's own VRAM either way).
#
# adsfactory talks to whichever instance it was assigned purely over HTTP
# (POST /prompt, /upload/image, GET /view, /system_stats) -- it never assumes
# a literal host path, so per-instance directories are fully transparent to it.
set -euo pipefail

INSTANCES="${COMFY_INSTANCES:-1}"
BASE_PORT="${COMFY_BASE_PORT:-8188}"
# Same flags the single-instance CMD always ran with (Dockerfile.runpod, the
# public repo's pre-existing default) -- override via COMFY_EXTRA_ARGS if a
# future round needs different per-deployment tuning without another image change.
EXTRA_ARGS="${COMFY_EXTRA_ARGS:---cache-lru 2 --reserve-vram 1}"

if ! [[ "$INSTANCES" =~ ^[0-9]+$ ]] || [ "$INSTANCES" -lt 1 ]; then
    echo "entrypoint: COMFY_INSTANCES=${INSTANCES} is not a positive integer -- defaulting to 1" >&2
    INSTANCES=1
fi

echo "entrypoint: starting ${INSTANCES} ComfyUI instance(s) on port(s) ${BASE_PORT}..$((BASE_PORT + INSTANCES - 1))"

pids=()
for i in $(seq 0 $((INSTANCES - 1))); do
    port=$((BASE_PORT + i))
    inst_dir="/ComfyUI/instances/${i}"
    mkdir -p "${inst_dir}/input" "${inst_dir}/output" "${inst_dir}/user" "${inst_dir}/temp"
    # shellcheck disable=SC2086
    python3 main.py --listen 0.0.0.0 --port "${port}" \
        --input-directory "${inst_dir}/input" \
        --output-directory "${inst_dir}/output" \
        --user-directory "${inst_dir}/user" \
        --temp-directory "${inst_dir}/temp" \
        ${EXTRA_ARGS} &
    pid=$!
    pids+=("${pid}")
    echo "entrypoint: instance ${i} started on port ${port} (pid ${pid}, dirs under ${inst_dir})"
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
