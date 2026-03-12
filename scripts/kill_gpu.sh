#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/kill_gpu.sh [options]

Kill active GPU compute processes reported by nvidia-smi.

Options:
  -n, --dry-run         Show matching PIDs but do not kill
  -s, --signal SIGNAL   Signal passed to kill (default: -9)
  -h, --help            Show this help

Examples:
  ./scripts/kill_gpu.sh
  ./scripts/kill_gpu.sh --dry-run
  ./scripts/kill_gpu.sh --signal -15
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

collect_gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF { print $1 }' \
    | sort -u
}

show_gpu_processes() {
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
}

dry_run=0
signal_opt="-9"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run)
      dry_run=1
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

require_cmd nvidia-smi
require_cmd awk
require_cmd sort
require_cmd kill

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
    echo "Killed PID ${pid} with signal ${signal_opt}"
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
  echo "Remaining process details:"
  show_gpu_processes
fi
