#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGTUNE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$FLAGTUNE_DIR/.." && pwd)"

MODEL="Qwen3.5-35B-A3B-p32768d1024"
YAML_NAME="Qwen3.5-35B-A3B-p32768d1024"
MODEL_SPECIFIED=false
YAML_SPECIFIED=false
OP="mm"
CACHE_DIR="/root/.flaggems"
DTYPES="bfloat16"
WARMUP=100
PARALLEL=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      MODEL_SPECIFIED=true
      shift 2
      ;;
    --yaml)
      YAML_NAME="$2"
      YAML_SPECIFIED=true
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
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--model <model_name>] [--yaml <yaml_name>] [--op <op_name>] [--cache-dir <dir>] [--dtypes <dtype_list>] [--warmup <count>] [--parallel <count>]"
      echo "Behavior:"
      echo "  - If --model is provided, use model mode and run shape-gen.py."
      echo "  - Else if --yaml is provided, use yaml mode and load FlagTune/shape-config/<yaml_name>.yaml directly."
      echo "  - If neither --model nor --yaml is provided, default to model mode."
      echo "Defaults:"
      echo "  - model: Qwen3.5-35B-A3B-p32768d1024"
      echo "  - yaml: Qwen3.5-35B-A3B-p32768d1024"
      echo "  - op: mm"
      echo "  - cache-dir: /root/.flaggems"
      echo "  - dtypes: bfloat16"
      echo "  - warmup: 100"
      echo "  - parallel: 8"
      echo "Default run: $0 --model Qwen3.5-35B-A3B-p32768d1024 --op mm --cache-dir /root/.flaggems --dtypes bfloat16 --warmup 100 --parallel 8"
      echo "Example 1: $0 --model Qwen3.5-35B-A3B-p32768d1024 --op mm"
      echo "Example 2: $0 --yaml Qwen3.5-397B-A17B-p32768d1024 --op mm"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

YAML_NAME="${YAML_NAME%.yaml}"
SHAPE_CONFIG_DIR="$FLAGTUNE_DIR/shape-config"

if [[ "${MODEL_SPECIFIED,,}" == "true" ]]; then
  RUN_NAME="$MODEL"
  SHAPE_FILE="$SHAPE_CONFIG_DIR/${MODEL}.yaml"
  SHOULD_GENERATE_SHAPES=true
elif [[ "${YAML_SPECIFIED,,}" == "true" ]]; then
  RUN_NAME="$YAML_NAME"
  SHAPE_FILE="$SHAPE_CONFIG_DIR/${YAML_NAME}.yaml"
  SHOULD_GENERATE_SHAPES=false
else
  RUN_NAME="$MODEL"
  SHAPE_FILE="$SHAPE_CONFIG_DIR/${MODEL}.yaml"
  SHOULD_GENERATE_SHAPES=true
fi

if [[ -n "$OP" ]]; then
  LOG_DIR="$REPO_ROOT/log/flagtune/${RUN_NAME}/${OP}/pretune"
else
  LOG_DIR="$REPO_ROOT/log/flagtune/${RUN_NAME}/pretune"
fi
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/pretune.log"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$REPO_ROOT"

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
    benchmark/test_blas_perf_parallel.py
    -m "$OP"
    -s
    --shape_file "$shape_file"
    --level core
    --mode kernel
    --parallel "$PARALLEL"
    --warmup "$WARMUP"
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

REPORT_MD="$FLAGTUNE_DIR/reports/${RUN_NAME}_${OP}.md"
REPORT_XLSX="$FLAGTUNE_DIR/reports/${RUN_NAME}_${OP}.xlsx"
REPORT_GENERATED=false

echo "[INFO] Flagtune started."

if [[ "${SHOULD_GENERATE_SHAPES,,}" == "true" ]]; then
  echo "[INFO] Using --model mode: generating shape config for ${MODEL}."
  run_cmd python "$FLAGTUNE_DIR/processing/shape-gen.py" --model "$MODEL"
else
  echo "[INFO] Using --yaml mode: skipping shape-gen.py and loading ${SHAPE_FILE}."
fi

if [[ ! -f "$SHAPE_FILE" ]]; then
  echo "[ERROR] Shape yaml not found: $SHAPE_FILE"
  exit 1
fi

if is_mm_family_op "$OP"; then
  run_mm_benchmark "default" "$SHAPE_FILE" false
  run_mm_benchmark "expand" "$SHAPE_FILE" false

  echo "[INFO] Generating pretune report."
  run_cmd python3 "$FLAGTUNE_DIR/processing/summary.py" --model "$RUN_NAME" --op "$OP"
  REPORT_GENERATED=true

fi


echo

echo "[DONE] Flagtune finished successfully."
echo "[LOG] Output Log saved to $LOG_FILE."
if [[ "${REPORT_GENERATED,,}" == "true" ]]; then
  echo "[REPORT] Markdown report saved to $REPORT_MD."
  echo "[REPORT] Excel report saved to $REPORT_XLSX."
fi
