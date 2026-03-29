#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen3.5-35B-A3B"
OP="mm"
CACHE_DIR="/root/.flaggems"
DTYPES="bfloat16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
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
    -h|--help)
      echo "Usage: $0 [--model <model_name>] [--op <op_name>] [--cache-dir <dir>] [--dtypes <dtype_list>]"
      echo "Default: $0 --model Qwen3.5-35B-A3B --op mm --cache-dir /root/.flaggems --dtypes bfloat16"
      echo "Example: $0 --model Qwen3.5-35B-A3B --op mm --cache-dir /root/.flaggems --dtypes bfloat16"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -n "$OP" ]]; then
  LOG_DIR="./log/flagtune/${MODEL}/${OP}/pretune"
else
  LOG_DIR="./log/flagtune/${MODEL}/pretune"
fi
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/pretune.log"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

run_cmd() {
  echo
  echo "[RUN] $*"
  "$@"
}

print_stage_banner() {
  local stage_name="$1"
  echo "[INFO] ============================= pretuning ${OP} operator with ${stage_name} configuration.============================="
}

clear_flaggems_cache() {
  echo "[INFO] Deleting Flaggems cache $CACHE_DIR."
  run_cmd rm -rf "$CACHE_DIR"
}

run_mm_benchmark() {
  local stage_name="$1"
  local shape_file="$2"
  local need_clear_cache="${3:-true}"
  local -a pytest_args=(
    benchmark/test_blas_perf.py
    -m "$OP"
    -s
    --shape_file "$shape_file"
    --level core
    --mode kernel
    --parallel 8
    -v
  )

  if [[ "$OP" == "mm" ]]; then
    pytest_args+=(--dtypes "$DTYPES")
  fi

  print_stage_banner "$stage_name"
  if [[ "${need_clear_cache,,}" == "true" ]]; then
    clear_flaggems_cache
  fi

  if [[ "${stage_name,,}" == "expand" ]]; then
    run_cmd env USE_FLAGTUNE=1 pytest "${pytest_args[@]}"
  else
    run_cmd pytest "${pytest_args[@]}"
  fi
}

is_mm_family_op() {
  case "$1" in
    mm|w8a8_block_fp8_matmul|w8a8_block_fp8_matmul_deepgemm)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

REPORT_MD="flagtune/reports/${MODEL}_${OP}.md"
REPORT_XLSX="flagtune/reports/${MODEL}_${OP}.xlsx"

echo "[INFO] Flagtune started."

run_cmd python flagtune/processing/shape-gen.py --model "$MODEL"
if is_mm_family_op "$OP"; then
  run_mm_benchmark "default" "flagtune/shape-config/${MODEL}.yaml" false
  run_mm_benchmark "expand" "flagtune/shape-config/${MODEL}.yaml" false

  echo "[INFO] Generating pretune report."
  run_cmd python3 flagtune/processing/summary.py --model "$MODEL" --op "$OP"

fi


echo

echo "[DONE] Flagtune finished successfully."
echo "[LOG] Output Log saved to $LOG_FILE."
echo "[REPORT] Markdown report saved to $REPORT_MD."
echo "[REPORT] Excel report saved to $REPORT_XLSX."
