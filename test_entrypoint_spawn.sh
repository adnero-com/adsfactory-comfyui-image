#!/usr/bin/env bash
# INTEGRATION test for entrypoint.sh: it RUNS the real entrypoint and asserts
# what the old unit tests never could -- that each spawned process gets the
# complete, correct argv AND actually LISTENS on a reachable address.
#
#   sudo -E ./test_entrypoint_spawn.sh          # needs to create /ComfyUI
#   docker run --rm -v "$PWD:/w" -w /w ubuntu:24.04 bash ./test_entrypoint_spawn.sh
#
# WHY THIS FILE EXISTS. On 2026-09-07 pod bet287qx74j8v5 (2 x H100) ran an
# entrypoint whose unit tests all passed. Both ComfyUI instances started, logged
# a normal startup banner with correct per-instance directories, and then never
# bound a port: the worker polled /system_stats on 8188 and 8189 every 5s for
# the full 900s boot window and got 404 the whole way. The unit tests asserted
# the instance -> port -> GPU MAPPING; nothing asserted the argv that mapping
# turns into, and nothing asserted that a socket ever opened. A test suite that
# cannot fail while the container is unreachable is not guarding the thing that
# matters.
#
# So this harness replaces `python3` on PATH with a stub that (a) records its
# FULL argv and its CUDA_VISIBLE_DEVICES, and (b) binds the address and port
# taken from its OWN argv -- exactly like ComfyUI, which defaults to 127.0.0.1
# when --listen is absent. Then it asserts, on the real processes:
#
#   * every instance's argv, token for token, against the production baseline;
#   * every instance's CUDA_VISIBLE_DEVICES pin;
#   * that every instance's port is CONNECTABLE ON A NON-LOOPBACK ADDRESS
#     (this is what a dropped/overridden --listen breaks, and the only check
#     that would have caught the RunPod-proxy 404s);
#   * that a device list CUDA rejects degrades to ONE UNPINNED instance that is
#     still reachable on 8188, instead of pinned instances that die at boot.
#
# No GPU, no ComfyUI, no torch required.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRYPOINT="${here}/entrypoint.sh"

pass=0
fail=0
expect() {  # expect <name> <expected> <actual>
    local name="$1" want="$2" got="$3"
    if [ "${want}" = "${got}" ]; then
        pass=$((pass + 1)); echo "ok   - ${name}"
    else
        fail=$((fail + 1)); echo "FAIL - ${name}"
        echo "         expected: [${want}]"
        echo "         actual:   [${got}]"
    fi
}

# --- preconditions -----------------------------------------------------------
# The entrypoint creates /ComfyUI/instances/<i>/... and runs `python3 main.py`
# from the image's WORKDIR. Both are asserted verbatim below, so the harness
# uses the REAL paths rather than a parameterised stand-in -- a test that
# rewrites the paths could not catch a path regression.
if ! mkdir -p /ComfyUI 2>/dev/null || [ ! -w /ComfyUI ]; then
    echo "SKIP-FATAL: this test must be able to create /ComfyUI (the image's WORKDIR)."
    echo "            Run it as root: sudo -E ./test_entrypoint_spawn.sh"
    echo "            or in a container: docker run --rm -v \"\$PWD:/w\" -w /w ubuntu:24.04 bash ./test_entrypoint_spawn.sh"
    exit 2
fi
REAL_PYTHON3="$(command -v python3 || true)"
if [ -z "${REAL_PYTHON3}" ]; then
    echo "SKIP-FATAL: python3 is required (the stub uses it to open a real socket)."
    exit 2
fi
export REAL_PYTHON3

SANDBOX="$(mktemp -d)"
cleanup() {
    local f pid
    [ -n "${EP_PID:-}" ] && kill "${EP_PID}" 2>/dev/null
    for f in "${SANDBOX}"/out/pid.*; do
        [ -e "${f}" ] || continue
        pid="$(cat "${f}" 2>/dev/null)"
        [ -n "${pid}" ] && kill -9 "${pid}" 2>/dev/null
    done
    rm -rf "${SANDBOX}"
    return 0
}
trap cleanup EXIT

# --- the stub ComfyUI --------------------------------------------------------
mkdir -p "${SANDBOX}/bin" "${SANDBOX}/out" "${SANDBOX}/cwd"
: > "${SANDBOX}/cwd/main.py"   # the real image has one; the stub never reads it

cat > "${SANDBOX}/bin/python3" <<'STUB'
#!/usr/bin/env bash
# Stub ComfyUI. Records the full argv + CUDA_VISIBLE_DEVICES it was given, then
# binds the address/port ITS OWN ARGV asked for -- defaulting to 127.0.0.1 when
# --listen is absent, exactly like ComfyUI. A dropped --listen therefore shows
# up here as "unreachable from a non-loopback address", which is the production
# failure mode, rather than as a silently-passing test.
addr="127.0.0.1"; port="0"; args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --listen) addr="${args[i+1]:-}" ;;
        --port)   port="${args[i+1]:-}" ;;
    esac
done
printf '%s\n' "$@" > "${COMFY_TEST_OUT}/argv.${port}"
printf '%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}" > "${COMFY_TEST_OUT}/cvd.${port}"
# `exec` keeps this PID, so the harness can reap the listener deterministically
# between cases (a leftover 0.0.0.0 listener would make the NEXT case's
# reachability check pass for the wrong reason -- the exact way a test lies).
printf '%s\n' "$$" > "${COMFY_TEST_OUT}/pid.${port}"
exec "${REAL_PYTHON3}" -c '
import socket, sys, time
addr, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((addr, port))
s.listen(8)
while True:
    time.sleep(3600)
' "${addr}" "${port}"
STUB
chmod +x "${SANDBOX}/bin/python3"

# --- helpers -----------------------------------------------------------------
# The routable address of this host. Connecting HERE (not 127.0.0.1) is the
# whole point: it is what the RunPod proxy does, and what a 127.0.0.1 bind fails.
host_ip() {
    local ip=""
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [ -n "${ip}" ] || ip="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
    printf '%s\n' "${ip}"
}
HOST_IP="$(host_ip)"

listening_on() {  # listening_on <ip> <port> -> "yes"/"no"
    local ip="$1" port="$2"
    if [ -z "${ip}" ]; then printf 'no-ip\n'; return 0; fi
    if (exec 3<>"/dev/tcp/${ip}/${port}") 2>/dev/null; then printf 'yes\n'; else printf 'no\n'; fi
}

run_entrypoint() {  # run_entrypoint <expected_argv_files...>  (env comes from the caller)
    rm -rf "${SANDBOX}/out"; mkdir -p "${SANDBOX}/out"
    rm -rf /ComfyUI/instances
    (
        cd "${SANDBOX}/cwd" || exit 1
        PATH="${SANDBOX}/bin:${PATH}" \
        COMFY_TEST_OUT="${SANDBOX}/out" \
        bash "${ENTRYPOINT}" > "${SANDBOX}/entrypoint.log" 2>&1
    ) &
    EP_PID=$!
    local waited=0
    while [ "${waited}" -lt 100 ]; do
        local all=1 f
        for f in "$@"; do [ -s "${SANDBOX}/out/${f}" ] || all=0; done
        [ "${all}" = "1" ] && break
        sleep 0.2; waited=$((waited + 1))
    done
    sleep 1   # let the stub reach bind()/listen()
}

stop_entrypoint() {
    local f pid p waited
    [ -n "${EP_PID:-}" ] && kill "${EP_PID}" 2>/dev/null
    for f in "${SANDBOX}"/out/pid.*; do
        [ -e "${f}" ] || continue
        pid="$(cat "${f}" 2>/dev/null)"
        [ -n "${pid}" ] && kill -9 "${pid}" 2>/dev/null
    done
    # Nothing may still hold a test port when the next case starts: a leftover
    # 0.0.0.0 listener would make the next reachability assertion pass for the
    # wrong reason.
    for p in 8188 8189 8190 8191; do
        waited=0
        while [ "$(listening_on "${HOST_IP}" "${p}")" = "yes" ] || \
              [ "$(listening_on 127.0.0.1 "${p}")" = "yes" ]; do
            waited=$((waited + 1))
            [ "${waited}" -gt 50 ] && { echo "WARN: port ${p} still held after stop_entrypoint" >&2; break; }
            sleep 0.2
        done
    done
    EP_PID=""
    return 0
}

recorded_argv() { tr '\n' ' ' < "${SANDBOX}/out/argv.$1" 2>/dev/null | sed 's/ $//'; }
recorded_cvd()  { cat "${SANDBOX}/out/cvd.$1" 2>/dev/null; }

# The production baseline, as sys.argv (starts at main.py) -- the exact list a
# healthy pod reports at GET /system_stats.
BASE_0="main.py --listen 0.0.0.0 --port 8188 --input-directory /ComfyUI/instances/0/input --output-directory /ComfyUI/instances/0/output --user-directory /ComfyUI/instances/0/user --temp-directory /ComfyUI/instances/0/temp --cache-lru 2 --reserve-vram 1"
BASE_1="main.py --listen 0.0.0.0 --port 8189 --input-directory /ComfyUI/instances/1/input --output-directory /ComfyUI/instances/1/output --user-directory /ComfyUI/instances/1/user --temp-directory /ComfyUI/instances/1/temp --cache-lru 2 --reserve-vram 1"

echo "== case 1: 2 instances, 2 verified GPUs (the 2xH100 pod) =="
COMFY_INSTANCES=2 COMFY_GPU_DEVICES="0,1" COMFY_CUDA_PROBE='exit 0' \
    run_entrypoint argv.8188 argv.8189

expect "instance 0 REAL argv == the production baseline" "${BASE_0}" "$(recorded_argv 8188)"
expect "instance 1 REAL argv == the production baseline (port + dirs only)" "${BASE_1}" "$(recorded_argv 8189)"
expect "instance 0 is pinned to its own verified GPU" "0" "$(recorded_cvd 8188)"
expect "instance 1 is pinned to its own verified GPU" "1" "$(recorded_cvd 8189)"
expect "instance 0 is REACHABLE on a non-loopback address" "yes" "$(listening_on "${HOST_IP}" 8188)"
expect "instance 1 is REACHABLE on a non-loopback address" "yes" "$(listening_on "${HOST_IP}" 8189)"
expect "per-instance directories really exist" "input output temp user" \
    "$(ls /ComfyUI/instances/1 2>/dev/null | sort | tr '\n' ' ' | sed 's/ $//')"
stop_entrypoint

echo "== case 2: COMFY_EXTRA_ARGS must not be able to unbind the pod =="
COMFY_INSTANCES=1 COMFY_GPU_DEVICES="0" COMFY_CUDA_PROBE='exit 0' \
    COMFY_EXTRA_ARGS="--listen 127.0.0.1 --cache-lru 2 --reserve-vram 1" \
    run_entrypoint argv.8188

expect "an admin's --listen 127.0.0.1 is stripped from the REAL argv" "${BASE_0}" "$(recorded_argv 8188)"
expect "the instance is still REACHABLE on a non-loopback address" "yes" "$(listening_on "${HOST_IP}" 8188)"
stop_entrypoint

echo "== case 3: THE REGRESSION -- device tokens CUDA cannot open =="
# pod bet287qx74j8v5: two tokens were handed out, both ComfyUI processes died
# inside `import execution`, the container exited and 8188/8189 404'd for 900s.
# The pod must degrade to one UNPINNED, REACHABLE instance instead.
COMFY_INSTANCES=2 COMFY_GPU_DEVICES="4,7" COMFY_CUDA_PROBE='exit 1' \
    run_entrypoint argv.8188

expect "an unverifiable device list yields exactly ONE instance" "1" \
    "$(ls "${SANDBOX}/out" 2>/dev/null | grep -c '^argv\.')"
expect "that instance is UNPINNED (no dead CUDA_VISIBLE_DEVICES is exported)" "<unset>" "$(recorded_cvd 8188)"
expect "its argv is still the production baseline" "${BASE_0}" "$(recorded_argv 8188)"
expect "and 8188 is STILL REACHABLE -- reachability is never traded for a pin" "yes" \
    "$(listening_on "${HOST_IP}" 8188)"
case "$(cat "${SANDBOX}/entrypoint.log" 2>/dev/null)" in
    *"NOT usable by CUDA"*) expect "the dropped tokens are logged loudly" "yes" "yes" ;;
    *) expect "the dropped tokens are logged loudly" "yes" "nothing in the entrypoint log" ;;
esac
stop_entrypoint

echo "== case 4: one good token, one dead one =="
COMFY_INSTANCES=2 COMFY_GPU_DEVICES="0,7" \
    COMFY_CUDA_PROBE='[ "${CUDA_VISIBLE_DEVICES}" = "0" ]' \
    run_entrypoint argv.8188

expect "only the verified GPU gets an instance" "1" \
    "$(ls "${SANDBOX}/out" 2>/dev/null | grep -c '^argv\.')"
expect "and it is pinned to the token that PASSED verification" "0" "$(recorded_cvd 8188)"
expect "still reachable on a non-loopback address" "yes" "$(listening_on "${HOST_IP}" 8188)"
stop_entrypoint

echo
echo "${pass} passed, ${fail} failed"
[ "${fail}" -eq 0 ]
