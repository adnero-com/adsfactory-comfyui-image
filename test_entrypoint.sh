#!/usr/bin/env bash
# Unit test for entrypoint.sh's instance -> port -> GPU mapping (comfy_plan) and
# device-list discovery (comfy_gpu_devices). Runs anywhere: no GPU, no ComfyUI,
# no container -- entrypoint.sh is SOURCED with COMFY_ENTRYPOINT_SOURCE_ONLY=1,
# which defines the pure helpers and returns before starting anything.
#
#   ./test_entrypoint.sh
#
# The load-bearing case is the CLAMP: COMFY_INSTANCES greater than the visible
# GPU count must never spawn more instances than GPUs -- two ComfyUI processes
# holding Wan/InfiniteTalk BF16 on one 80GB card OOM.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=entrypoint.sh
COMFY_ENTRYPOINT_SOURCE_ONLY=1 source "${here}/entrypoint.sh"

pass=0
fail=0

expect() {  # expect <name> <expected> <actual>
    local name="$1" want="$2" got="$3"
    if [ "${want}" = "${got}" ]; then
        pass=$((pass + 1))
        echo "ok   - ${name}"
    else
        fail=$((fail + 1))
        echo "FAIL - ${name}"
        echo "         expected: [${want}]"
        echo "         actual:   [${got}]"
    fi
}

plan() {  # plan <instances> <base> <devices> -> plan lines, ';'-joined, stderr dropped
    comfy_plan "$1" "$2" "$3" 2>/dev/null | paste -sd';' -
}

warnings() {  # warnings <instances> <base> <devices> -> stderr only
    comfy_plan "$1" "$2" "$3" 2>&1 >/dev/null
}

# --- the single-GPU / single-instance case must stay exactly as it was --------
expect "1 instance, 1 GPU -> instance 0 on 8188 pinned to GPU 0" \
    "0 8188 0" "$(plan 1 8188 "0")"

# --- the 2xH100 pod this whole change exists for ------------------------------
expect "2 instances, 2 GPUs -> one instance per GPU, ports 8188/8189" \
    "0 8188 0;1 8189 1" "$(plan 2 8188 "0,1")"

expect "4 instances, 4 GPUs -> 4 pinned instances on 4 ports" \
    "0 8188 0;1 8189 1;2 8190 2;3 8191 3" "$(plan 4 8188 "0,1,2,3")"

# --- the OOM footgun: never hand out more instances than GPUs -----------------
expect "3 instances, 2 GPUs -> CLAMPED to 2" \
    "0 8188 0;1 8189 1" "$(plan 3 8188 "0,1")"

expect "2 instances, 1 GPU -> CLAMPED to 1" \
    "0 8188 0" "$(plan 2 8188 "0")"

case "$(warnings 3 8188 "0,1")" in
    *"CLAMPING to 2"*) expect "clamp is logged loudly" "yes" "yes" ;;
    *) expect "clamp is logged loudly" "yes" "no warning on stderr" ;;
esac

# --- no GPU visible at all (CPU-only builder/test container) ------------------
expect "1 instance, no GPU list -> unpinned, legacy behaviour" \
    "0 8188 -" "$(plan 1 8188 "")"

expect "2 instances, no GPU list -> CLAMPED to 1 unpinned instance" \
    "0 8188 -" "$(plan 2 8188 "")"

# --- malformed COMFY_INSTANCES ------------------------------------------------
expect "COMFY_INSTANCES=0 -> 1 instance" "0 8188 0" "$(plan 0 8188 "0,1")"
expect "COMFY_INSTANCES=abc -> 1 instance" "0 8188 0" "$(plan abc 8188 "0,1")"
expect "COMFY_INSTANCES=-2 -> 1 instance" "0 8188 0" "$(plan -2 8188 "0,1")"

# --- base-port override + device-token shapes ---------------------------------
expect "COMFY_BASE_PORT is honoured" \
    "0 9000 0;1 9001 1" "$(plan 2 9000 "0,1")"

expect "whitespace in the device list is trimmed" \
    "0 8188 0;1 8189 1" "$(plan 2 8188 " 0 , 1 ")"

expect "GPU UUIDs are handed out verbatim" \
    "0 8188 GPU-aaa;1 8189 GPU-bbb" "$(plan 2 8188 "GPU-aaa,GPU-bbb")"

expect "fewer instances than GPUs leaves the extra card idle (no error)" \
    "0 8188 0" "$(plan 1 8188 "0,1")"

# --- comfy_gpu_devices precedence --------------------------------------------
expect "COMFY_GPU_DEVICES wins over CUDA_VISIBLE_DEVICES" \
    "2,3" "$(COMFY_GPU_DEVICES=2,3 CUDA_VISIBLE_DEVICES=0,1 comfy_gpu_devices)"

expect "an inherited CUDA_VISIBLE_DEVICES is handed out as-is" \
    "0,1" "$(CUDA_VISIBLE_DEVICES=0,1 comfy_gpu_devices)"

expect "an inherited CUDA_VISIBLE_DEVICES restricted to one card yields one device" \
    "3" "$(CUDA_VISIBLE_DEVICES=3 comfy_gpu_devices)"

expect "instances are pinned to the INHERITED device tokens, not raw 0..N-1" \
    "0 8188 4;1 8189 7" "$(plan 2 8188 "$(CUDA_VISIBLE_DEVICES=4,7 comfy_gpu_devices)")"

expect "the NVIDIA_VISIBLE_DEVICES keyword 'all' is not a device list" \
    "" "$(CUDA_VISIBLE_DEVICES=all PATH=/nonexistent comfy_gpu_devices)"

expect "no nvidia-smi and no env -> empty device list" \
    "" "$(PATH=/nonexistent comfy_gpu_devices)"

echo
echo "${pass} passed, ${fail} failed"
[ "${fail}" -eq 0 ]
