#!/usr/bin/env bash
# Unit test for entrypoint.sh's PURE helpers: the COMPLETE per-instance ComfyUI
# argv (comfy_argv), the device-token verification that stands between a
# candidate list and a running process (comfy_verify_devices), the
# instance -> port -> GPU mapping (comfy_plan) and device-list discovery
# (comfy_gpu_devices). Runs anywhere: no GPU, no ComfyUI, no container --
# entrypoint.sh is SOURCED with COMFY_ENTRYPOINT_SOURCE_ONLY=1, which defines
# the pure helpers and returns before starting anything.
#
#   ./test_entrypoint.sh
#
# Two load-bearing groups:
#
#   * THE ARGV. The 2026-09-07 outage (pod bet287qx74j8v5) was a pod whose
#     ComfyUI processes started, logged a healthy banner, and never bound a
#     port. The old test suite passed throughout, because it asserted the
#     instance->GPU MAPPING and never the argv/environment the mapping actually
#     turns into. So: every instance's argv is now asserted in full, token for
#     token, against the argv a HEALTHY PRODUCTION POD reports for itself at
#     GET /system_stats -- --listen 0.0.0.0 and the per-instance --port
#     included. Nothing about how an instance is reachable is left implicit.
#
#   * THE CLAMP + THE VERIFY. COMFY_INSTANCES greater than the usable GPU count
#     must never spawn more instances than GPUs (two ComfyUI processes holding
#     Wan/InfiniteTalk BF16 on one 80GB card OOM), and a device token CUDA
#     cannot open must never reach a ComfyUI child -- it must be dropped, and if
#     nothing survives we run ONE UNPINNED instance rather than a pinned one
#     that dies in `import execution` and takes the pod's ports with it.
#
# See also test_entrypoint_spawn.sh, which asserts the same argv on REAL spawned
# processes and that each one actually LISTENS on its port.
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

argv() {  # argv <index> <port> [extra] -> the full argv, space-joined
    comfy_argv "$1" "$2" "${3-}" 2>/dev/null | paste -sd' ' -
}

# The historical per-instance flags, i.e. entrypoint.sh's COMFY_EXTRA_ARGS default.
DEFAULT_EXTRA="--cache-lru 2 --reserve-vram 1"

# ============================================================================
# 1. THE COMPLETE ARGV -- frozen against a healthy production pod
# ============================================================================
#
# Captured live from a RUNNING, RENDERING pod on the last-known-good image
# (digest sha256:0c50488...) via GET /system_stats -> .system.argv:
#
#   ["main.py","--listen","0.0.0.0","--port","8188",
#    "--input-directory","/ComfyUI/instances/0/input",
#    "--output-directory","/ComfyUI/instances/0/output",
#    "--user-directory","/ComfyUI/instances/0/user",
#    "--temp-directory","/ComfyUI/instances/0/temp",
#    "--cache-lru","2","--reserve-vram","1"]
#
# (/system_stats reports sys.argv, which starts at main.py; we assert the
# interpreter too.) If ANY token here changes, a render pod's reachability or
# its on-disk isolation changed with it -- that is a deliberate decision, never
# a refactor side effect.
BASELINE_0="python3 main.py --listen 0.0.0.0 --port 8188 --input-directory /ComfyUI/instances/0/input --output-directory /ComfyUI/instances/0/output --user-directory /ComfyUI/instances/0/user --temp-directory /ComfyUI/instances/0/temp --cache-lru 2 --reserve-vram 1"
BASELINE_1="python3 main.py --listen 0.0.0.0 --port 8189 --input-directory /ComfyUI/instances/1/input --output-directory /ComfyUI/instances/1/output --user-directory /ComfyUI/instances/1/user --temp-directory /ComfyUI/instances/1/temp --cache-lru 2 --reserve-vram 1"

expect "instance 0 argv is byte-identical to the production baseline" \
    "${BASELINE_0}" "$(argv 0 8188 "${DEFAULT_EXTRA}")"

expect "instance 1 argv differs from instance 0 ONLY in port + instance dir" \
    "${BASELINE_1}" "$(argv 1 8189 "${DEFAULT_EXTRA}")"

# The single most important token in the whole file: without it ComfyUI binds
# 127.0.0.1 and the RunPod proxy can never reach it, while the process looks
# perfectly healthy in the log.
expect "every instance listens on 0.0.0.0 (never the ComfyUI 127.0.0.1 default)" \
    "--listen 0.0.0.0" \
    "$(comfy_argv 3 8191 "${DEFAULT_EXTRA}" | grep -A1 -x -- '--listen' | paste -sd' ' -)"

expect "the port is the instance's OWN port, once" \
    "1" "$(comfy_argv 2 8190 "${DEFAULT_EXTRA}" | grep -c -x -- '--port')"

expect "instance 2 is told port 8190" \
    "--port 8190" \
    "$(comfy_argv 2 8190 "${DEFAULT_EXTRA}" | grep -A1 -x -- '--port' | paste -sd' ' -)"

expect "the argv is exactly 18 separate tokens (never quoted into one blob)" \
    "18" "$(comfy_argv 0 8188 "${DEFAULT_EXTRA}" | wc -l | tr -d ' ')"

expect "COMFY_EXTRA_ARGS is word-split and appended verbatim" \
    "python3 main.py --listen 0.0.0.0 --port 8188 --input-directory /ComfyUI/instances/0/input --output-directory /ComfyUI/instances/0/output --user-directory /ComfyUI/instances/0/user --temp-directory /ComfyUI/instances/0/temp --fast --highvram" \
    "$(argv 0 8188 "--fast --highvram")"

expect "no COMFY_EXTRA_ARGS at all still yields a complete, reachable argv" \
    "python3 main.py --listen 0.0.0.0 --port 8188 --input-directory /ComfyUI/instances/0/input --output-directory /ComfyUI/instances/0/output --user-directory /ComfyUI/instances/0/user --temp-directory /ComfyUI/instances/0/temp" \
    "$(argv 0 8188 "")"

# COMFY_EXTRA_ARGS is an admin string; it must not be able to unbind the pod.
expect "COMFY_EXTRA_ARGS cannot override --listen (ComfyUI lets the last flag win)" \
    "${BASELINE_0}" \
    "$(argv 0 8188 "--listen 127.0.0.1 --cache-lru 2 --reserve-vram 1")"

expect "COMFY_EXTRA_ARGS cannot override --port either" \
    "${BASELINE_0}" \
    "$(argv 0 8188 "--cache-lru 2 --reserve-vram 1 --port 9999")"

expect "the --listen=addr spelling is refused too" \
    "${BASELINE_0}" \
    "$(argv 0 8188 "--listen=127.0.0.1 --cache-lru 2 --reserve-vram 1")"

case "$(comfy_extra_args "--listen 127.0.0.1" 2>&1 >/dev/null)" in
    *"refusing '--listen'"*) expect "refusing an unbinding flag is logged loudly" "yes" "yes" ;;
    *) expect "refusing an unbinding flag is logged loudly" "yes" "no warning on stderr" ;;
esac

# ============================================================================
# 2. DEVICE VERIFICATION -- no unproven token ever pins a ComfyUI
# ============================================================================
# COMFY_CUDA_PROBE stands in for "ask CUDA whether it can open this device":
# exit 0 == usable. In prod it is a real torch probe; here it is a predicate.
verify() {  # verify <probe> <candidates_csv>
    COMFY_CUDA_PROBE="$1" comfy_verify_devices "$2" 2>/dev/null
}

ALL_OK='exit 0'
NONE_OK='exit 1'
ONLY_0='[ "${CUDA_VISIBLE_DEVICES}" = "0" ]'

expect "a fully usable device list survives verification unchanged" \
    "0,1" "$(verify "${ALL_OK}" "0,1")"

expect "the probe sees the EXACT token as CUDA_VISIBLE_DEVICES" \
    "0" "$(verify "${ONLY_0}" "0,1")"

expect "a device list CUDA rejects verifies to nothing" \
    "" "$(verify "${NONE_OK}" "0,1")"

expect "verification preserves order and drops only the dead tokens" \
    "GPU-aaa,GPU-ccc" \
    "$(verify '[ "${CUDA_VISIBLE_DEVICES}" != "GPU-bbb" ]' "GPU-aaa,GPU-bbb,GPU-ccc")"

case "$(COMFY_CUDA_PROBE="${NONE_OK}" comfy_verify_devices "0,1" 2>&1 >/dev/null)" in
    *"NOT usable by CUDA"*) expect "a dropped device token is logged loudly" "yes" "yes" ;;
    *) expect "a dropped device token is logged loudly" "yes" "no warning on stderr" ;;
esac

# THE REGRESSION ITSELF (pod bet287qx74j8v5): a candidate list that CUDA cannot
# open must degrade to ONE UNPINNED instance -- reachable on 8188 -- and must
# NEVER produce a pinned instance that dies inside `import execution`.
expect "2 instances whose ONLY candidates fail CUDA -> 1 UNPINNED instance, still on 8188" \
    "0 8188 -" "$(plan 2 8188 "$(verify "${NONE_OK}" "0,1")")"

expect "2 instances, only ONE token verifies -> 1 instance PINNED to that token" \
    "0 8188 0" "$(plan 2 8188 "$(verify "${ONLY_0}" "0,1")")"

expect "2 instances, both tokens verify -> the 2xH100 pod this change exists for" \
    "0 8188 0;1 8189 1" "$(plan 2 8188 "$(verify "${ALL_OK}" "0,1")")"

expect "an unverifiable COMFY_GPU_DEVICES override is dropped like any other" \
    "" "$(COMFY_GPU_DEVICES=9,9 verify "${NONE_OK}" "$(COMFY_GPU_DEVICES=9,9 comfy_gpu_devices)")"

# ============================================================================
# 3. THE MAPPING (unchanged behaviour, still guarded)
# ============================================================================

# --- the single-GPU / single-instance case must stay exactly as it was --------
expect "1 instance, 1 GPU -> instance 0 on 8188 pinned to GPU 0" \
    "0 8188 0" "$(plan 1 8188 "0")"

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

expect "COMFY_BASE_PORT reaches the argv too" \
    "--port 9001" \
    "$(comfy_argv 1 9001 "${DEFAULT_EXTRA}" | grep -A1 -x -- '--port' | paste -sd' ' -)"

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
