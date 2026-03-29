#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGTUNE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SHAPE_CONFIG_DIR="$FLAGTUNE_DIR/shape-config"
PRETUNE_SCRIPT="$SCRIPT_DIR/pretune.sh"
OP="mm"
CACHE_DIR="/root/.flaggems"
DTYPES="bfloat16"
WARMUP=100
PARALLEL=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --op)
      OP="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --dtypes)
      DTYPES="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--op <op_name>] [--cache-dir <dir>] [--dtypes <dtype_list>] [--warmup <count>] [--parallel <count>]"
      echo "Default: $0 --op mm --cache-dir /root/.flaggems --dtypes bfloat16 --warmup 100 --parallel 8"
      echo "Example: $0 --op mm --cache-dir /root/.flaggems --dtypes bfloat16 --warmup 100 --parallel 8"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ! -d "$SHAPE_CONFIG_DIR" ]]; then
  echo "[ERROR] shape-config directory not found: $SHAPE_CONFIG_DIR"
  exit 1
fi

if [[ ! -x "$PRETUNE_SCRIPT" ]]; then
  if [[ -f "$PRETUNE_SCRIPT" ]]; then
    chmod +x "$PRETUNE_SCRIPT"
  else
    echo "[ERROR] pretune script not found: $PRETUNE_SCRIPT"
    exit 1
  fi
fi

mapfile -t shape_files < <(find "$SHAPE_CONFIG_DIR" -maxdepth 1 -type f -name "*.txt" | sort)

if [[ ${#shape_files[@]} -eq 0 ]]; then
  echo "[ERROR] No files found in: $SHAPE_CONFIG_DIR"
  exit 1
fi

total=${#shape_files[@]}
index=0
failed=0

echo "[INFO] Found $total shape-config files."

for file in "${shape_files[@]}"; do
  index=$((index + 1))
  model="$(basename "$file")"
  model="${model%.*}"

  echo
  echo "[INFO] [$index/$total] Running pretune for model: $model"
  if ! "$PRETUNE_SCRIPT" --model "$model" --op "$OP" --cache-dir "$CACHE_DIR" --dtypes "$DTYPES" --warmup "$WARMUP" --parallel "$PARALLEL"; then
    failed=$((failed + 1))
    echo "[ERROR] pretune failed for model: $model"
  fi
done

echo
if [[ $failed -gt 0 ]]; then
  echo "[DONE] Batch run finished with failures: $failed/$total"
  exit 1
fi

echo "[DONE] Batch run finished successfully: $total/$total"
