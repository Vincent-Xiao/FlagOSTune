#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./flagtune/kill-gpu.sh [options]

Kill active GPU compute processes reported by the vendor GPU tool.
Supported tools: nvidia-smi, mthreads-gmi.

Options:
  -n, --dry-run         Show matching PIDs but do not kill
  -g, --gpus COUNT      Only target the first COUNT GPUs (default: 8)
  -s, --signal SIGNAL   Signal passed to kill (default: -9)
  -h, --help            Show this help

Examples:
  ./flagtune/kill-gpu.sh
  ./flagtune/kill-gpu.sh --gpus 4
  ./flagtune/kill-gpu.sh --dry-run
  ./flagtune/kill-gpu.sh --signal -15
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

get_pid_stat() {
  local pid="$1"
  ps -o stat= -p "$pid" 2>/dev/null | awk 'NR == 1 { print $1 }'
}

get_pid_wchan() {
  local pid="$1"
  ps -o wchan= -p "$pid" 2>/dev/null | awk 'NR == 1 { print $1 }'
}

collect_gpu_pids() {
  case "$gpu_tool" in
    nvidia-smi)
      awk -F', *' -v limit="$gpu_count" '
        NR == FNR {
          if ($1 ~ /^[0-9]+$/ && $1 < limit) {
            target[$2] = 1
          }
          next
        }
        target[$1] && $2 ~ /^[0-9]+$/ {
          print $2
        }
      ' \
        <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null) \
        <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null) \
        | sort -u
      ;;
    mthreads-gmi)
      mthreads-gmi -pm 2>/dev/null \
        | awk -v limit="$gpu_count" '$1 ~ /^[0-9]+$/ && $1 < limit && $2 ~ /^[0-9]+$/ { print $2 }' \
        | sort -u
      ;;
    *)
      echo "Error: unsupported GPU tool: $gpu_tool" >&2
      exit 1
      ;;
  esac
}

show_gpu_processes() {
  case "$gpu_tool" in
    nvidia-smi)
      nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
      ;;
    mthreads-gmi)
      mthreads-gmi -pm 2>/dev/null || true
      ;;
    *)
      echo "Error: unsupported GPU tool: $gpu_tool" >&2
      exit 1
      ;;
  esac
}

detect_gpu_tool() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi"
    return
  fi
  if command -v mthreads-gmi >/dev/null 2>&1; then
    echo "mthreads-gmi"
    return
  fi
  echo "Error: no supported GPU tool found (expected nvidia-smi or mthreads-gmi)." >&2
  exit 1
}

dry_run=0
gpu_count=8
signal_opt="-9"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run)
      dry_run=1
      ;;
    -g|--gpus)
      if [[ $# -lt 2 ]]; then
        echo "Error: --gpus requires a value" >&2
        exit 1
      fi
      gpu_count="$2"
      shift
      ;;
    -s|--signal)
      if [[ $# -lt 2 ]]; then
        echo "Error: --signal requires a value" >&2
        exit 1
      fi
      signal_opt="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if ! [[ "$gpu_count" =~ ^[0-9]+$ ]] || [[ "$gpu_count" -le 0 ]]; then
  echo "Error: --gpus must be a positive integer" >&2
  exit 1
fi

require_cmd awk
require_cmd sort
require_cmd kill
require_cmd ps
gpu_tool="$(detect_gpu_tool)"

echo "Detected GPU tool: ${gpu_tool}"
echo "Target GPU range: [0, $((gpu_count - 1))]"
echo "Current GPU compute processes:"
show_gpu_processes

pids="$(collect_gpu_pids || true)"

if [[ -z "${pids}" ]]; then
  echo "No GPU compute processes found."
  exit 0
fi

echo "Target PIDs: ${pids}"

if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry-run mode: no process was killed."
  exit 0
fi

for pid in $pids; do
  if kill "$signal_opt" "$pid" 2>/dev/null; then
    echo "Sent signal ${signal_opt} to PID ${pid}"
  else
    echo "Warning: failed to kill PID ${pid}" >&2
  fi
done

sleep 1
remaining="$(collect_gpu_pids || true)"

if [[ -z "${remaining}" ]]; then
  echo "All GPU compute processes were cleared."
else
  echo "Remaining GPU compute PIDs: ${remaining}"
  for pid in $remaining; do
    if kill -0 "$pid" 2>/dev/null; then
      stat="$(get_pid_stat "$pid")"
      wchan="$(get_pid_wchan "$pid")"
      if [[ "${stat}" == D* ]]; then
        echo "Warning: PID ${pid} is stuck in uninterruptible sleep (STAT=${stat}, WCHAN=${wchan:-unknown}). SIGKILL cannot remove it until the kernel/driver wait completes." >&2
      else
        echo "Warning: PID ${pid} is still alive after signal ${signal_opt} (STAT=${stat:-unknown}, WCHAN=${wchan:-unknown})." >&2
      fi
    else
      echo "Warning: PID ${pid} is no longer alive, but it is still reported by ${gpu_tool}. The vendor tool output may be stale." >&2
    fi
  done
  echo "Remaining process details:"
  show_gpu_processes
fi
