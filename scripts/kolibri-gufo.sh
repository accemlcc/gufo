#!/usr/bin/env bash
# kolibri-gufo.sh - start, stop and watch a served Kolibri-1 on one Strix Halo.
#
# Kolibri-1 at --context 131072 holds about 57.5 GiB of device memory: 45 GiB of
# weights plus the cache of the ten patternless layers. On a 128 GiB UMA box
# that is nearly the whole budget, so nothing else large may sit on the device
# while this runs. The script gates on measured state - host memory and GTT in
# use - and names whatever holds the device instead of stopping something by
# guesswork. It also does not report ready until /health answers, so a caller
# never sends a request to a model that is still loading.
#
# Usage: kolibri-gufo.sh start|stop|status|logs [n]
# Environment:
#   KOLIBRI_BIN   server binary   (default build/gpu-test/gufo)
#   KOLIBRI_MODEL model file      (default the Q4_K_M conversion)
#   KOLIBRI_PORT  port            (default 8102)
#   KOLIBRI_CTX   context         (default 131072)
#   KOLIBRI_LOG   log file        (default ~/kolibri-gufo-server.log)
#   KOLIBRI_PEER  optional container to stop first (default: none)
#   KOLIBRI_MIN_AVAILABLE  memory floor in GiB (default 16)
#   KOLIBRI_ROCM_LIB  ROCm userspace to run against
#                     (default ~/rocm-sdk/lib, the release the build used)
set -euo pipefail

BIN="${KOLIBRI_BIN:-$HOME/gufo-kolibri-src/build/gpu-test/gufo}"
MODEL="${KOLIBRI_MODEL:-$HOME/models/Kolibri-1-GGUF/Kolibri-1-Q4_K_M.gguf}"
PORT="${KOLIBRI_PORT:-8102}"
CTX="${KOLIBRI_CTX:-131072}"
LOG="${KOLIBRI_LOG:-$HOME/kolibri-gufo-server.log}"
PEER="${KOLIBRI_PEER:-}"
PIDFILE="${KOLIBRI_PIDFILE:-$HOME/kolibri-gufo-server.pid}"
# Rule 3 of the box: nothing large starts under 16 GiB available. A load that
# swaps does not just fail slowly - the OOM killer takes the agent session with
# it, and a wedged ROCm context needs a reboot. The floor is an environment
# knob so the gate itself can be tested without emptying the machine first.
MIN_AVAILABLE_GiB="${KOLIBRI_MIN_AVAILABLE:-16}"

# The ROCm userspace the binary is run against, named instead of inherited.
# Kolibri loads no BLAS at all, so this is only ever libamdhip64: the same
# binary under the host's own 7.2 line schedules the device differently, and a
# long prompt stops repeating bit for bit (docs/models/kolibri-1/EXPERIMENTS.md,
# entry 2026-10-15). Serving on the build release is what makes the number in
# the log the number the engine was measured with, so a missing runtime stops
# the start rather than falling back to the host's.
ROCM_LIB="${KOLIBRI_ROCM_LIB:-$HOME/rocm-sdk/lib}"

# The version of the libamdhip64 in a directory. The SDK keeps its own file
# under the bare soname and points the versioned name at it, so `readlink -f`
# would answer "libamdhip64.so.7" for both arms; the versioned sibling is what
# says which release a directory holds.
runtime_version() {
  local candidate
  for candidate in "$1"/libamdhip64.so.7.*; do
    [[ -e "$candidate" ]] || continue
    printf '%s\n' "${candidate##*/libamdhip64.so.}"
    return
  done
  printf -- '-\n'
}

# The libamdhip64 a running process really mapped, as version and directory.
loaded_runtime() {
  local mapped dir
  mapped="$(sed -n 's#.* \(/[^ ]*libamdhip64\.so\.[0-9].*\)#\1#p' \
    "/proc/$1/maps" 2>/dev/null | head -1)"
  [[ -n "$mapped" ]] || { printf -- '-\n'; return; }
  dir="$(dirname "$mapped")"
  printf '%s %s\n' "$(runtime_version "$dir")" "$dir"
}

available_gib() {
  awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo
}

# GTT is where the weights live on an APU. The `vram` counters are a small
# carve-out (1 GiB here) and read as full even when idle, so they say nothing
# about capacity.
gtt_used_gib() {
  awk '{s += $1} END {printf "%d", s / 1073741824}' \
    /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null || echo 0
}

device_holders() {
  if command -v amd-smi >/dev/null; then
    amd-smi process 2>/dev/null | sed -n '1,12p'
  else
    pgrep -af 'gufo|llama-server|vllm' | cut -c1-110 || true
  fi
}

running_pid() {
  local pid=""
  [[ -f "$PIDFILE" ]] && pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  # setsid only execs when the caller is not a process-group leader, so `$!`
  # can be a wrapper that exits right away. Fall back to the binary itself,
  # otherwise a healthy server looks dead while its weights are still loading.
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    pid="$(pgrep -f -- "${BIN} serve llm --model ${MODEL}" | head -1 || true)"
    [[ -n "$pid" ]] && printf '%s\n' "$pid" > "$PIDFILE"
  fi
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && printf '%s\n' "$pid"
}

health() {
  curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}
stop_server() {
  local pid
  if pid="$(running_pid)"; then
    kill "$pid"
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    echo "server gestoppt (pid $pid)"
  fi
  rm -f "$PIDFILE"
}

case "${1:-status}" in
  start)
    if health; then
      echo "8102 antwortet bereits; erst stoppen, dann neu starten" >&2
      exit 1
    fi
    [[ -x "$BIN" ]] || { echo "keine Binary: $BIN" >&2; exit 1; }
    [[ -f "$MODEL" ]] || { echo "kein Modell: $MODEL" >&2; exit 1; }
    if [[ ! -e "$ROCM_LIB/libamdhip64.so.7" ]]; then
      echo "Abbruch: kein libamdhip64 in $ROCM_LIB." >&2
      echo "Der Server laeuft gegen die ROCm, gegen die die Binary gebaut wurde;" >&2
      echo "die Userspace des Hosts ist eine andere Zahl und ein anderes" >&2
      echo "Ergebnis (docs/models/kolibri-1/EXPERIMENTS.md, 2026-10-15)." >&2
      echo "Andere Userspace: KOLIBRI_ROCM_LIB=... oder neu bauen." >&2
      exit 1
    fi
    avail="$(available_gib)"
    if (( avail < MIN_AVAILABLE_GiB )); then
      echo "Abbruch: ${avail} GiB verfuegbar, unter ${MIN_AVAILABLE_GiB} GiB startet" \
           "nichts Grosses (Rule 3)" >&2
      exit 1
    fi
    # The device gate. 8 GiB of GTT is far above an idle context and far below
    # one loaded model, so it says "something real is still resident" without
    # tripping on jitter. Which engine loses that space is the operator's call:
    # these boxes run several, so a container is only stopped when it was named.
    busy="$(gtt_used_gib)"
    if (( busy > 8 )); then
      echo "Abbruch: ${busy} GiB GTT belegt; erst das laufende Modell beenden." >&2
      device_holders >&2
      exit 1
    fi
    if [[ -n "$PEER" ]] && command -v podman >/dev/null &&
       podman ps -q -f "name=^${PEER}$" | grep -q .; then
      echo "stoppe ${PEER} (derselbe UMA-Speicher)"
      podman stop -t 20 "$PEER" >/dev/null
    fi
    echo "starte Kolibri-1 (Kontext ${CTX}, ${avail} GiB verfuegbar, runtime" \
         "$(runtime_version "$ROCM_LIB"))"
    : > "$LOG"
    echo "runtime: $ROCM_LIB -> $(runtime_version "$ROCM_LIB")" >> "$LOG"
    LD_LIBRARY_PATH="$ROCM_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    setsid nohup "$BIN" serve llm \
      --model "$MODEL" \
      --served-model-name Kolibri-1 \
      --context "$CTX" \
      --sessions 1 \
      --host 0.0.0.0 --port "$PORT" \
      >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    # Weights alone are 45 GiB; the load takes about 15 s, the first mmap pass
    # longer on a cold page cache. 180 s is generous, not a guess: it is what a
    # cold start on this box measured.
    for _ in $(seq 1 180); do
      health && { echo "bereit auf :${PORT} ($(available_gib) GiB frei)"; exit 0; }
      running_pid > /dev/null || { echo "Server ist gestorben:"; tail -20 "$LOG" >&2; exit 1; }
      sleep 1
    done
    echo "nach 180 s nicht bereit:"; tail -20 "$LOG" >&2
    exit 1
    ;;
  stop)
    stop_server
    echo "frei: $(available_gib) GiB verfuegbar"
    ;;
  status)
    if pid="$(running_pid)"; then
      printf 'pid %s, health: ' "$pid"
      health && echo ok || echo "antwortet nicht"
      printf 'runtime: %s\n' "$(loaded_runtime "$pid")"
      device_holders
    else
      echo "laeuft nicht"
    fi
    # Reported even when this server is down: the number that decides whether a
    # start is possible at all belongs to the device, not to the process.
    printf 'GTT benutzt: %s GiB, verfuegbar: %s GiB\n' \
      "$(gtt_used_gib)" "$(available_gib)"
    ;;
  logs)
    tail -n "${2:-60}" "$LOG"
    ;;
  *)
    echo "usage: $0 start|stop|status|logs [n]" >&2
    exit 2
    ;;
esac
