#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGTUNE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$FLAGTUNE_DIR/.." && pwd)"

MODEL="Qwen3.5-35B-A3B-p32768d1024"
YAML_NAME="Qwen3.5-35B-A3B-p32768d1024"
MODEL_SPECIFIED=false
YAML_SPECIFIED=false
BRANCH_COMPARE=false
COMPARE_BRANCH="master"
CLEAR_CACHE=false
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
    --branch)
      BRANCH_COMPARE=true
      if [[ $# -gt 1 && "$2" != --* ]]; then
        COMPARE_BRANCH="$2"
        shift 2
      else
        shift 1
      fi
      ;;
    --clear-cache)
      CLEAR_CACHE=true
      shift 1
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
      echo "Usage: $0 [--model <model_name>] [--yaml <yaml_name>] [--branch [branch_name]] [--clear-cache] [--op <op_name>] [--cache-dir <dir>] [--dtypes <dtype_list>] [--warmup <count>] [--parallel <count>]"
      echo "Behavior:"
      echo "  - If --model is provided, use model mode and run shape-gen.py."
      echo "  - Else if --yaml is provided, use yaml mode and load FlagTune/shape-config/<yaml_name>.yaml directly."
      echo "  - If neither --model nor --yaml is provided, default to model mode."
      echo "  - If --branch is provided, use git checkout in the current repo to compare the specified branch vs current branch using default configuration on both sides."
      echo "  - If --branch is provided without a value, the compare branch defaults to master."
      echo "  - If any git checkout fails, the script exits immediately and prints the git error."
      echo "  - If --clear-cache is provided, delete Flaggems cache before each benchmark run."
      echo "Defaults:"
      echo "  - model: Qwen3.5-35B-A3B-p32768d1024"
      echo "  - yaml: Qwen3.5-35B-A3B-p32768d1024"
      echo "  - op: mm"
      echo "  - cache-dir: /root/.flaggems"
      echo "  - dtypes: bfloat16"
      echo "  - warmup: 100"
      echo "  - parallel: 8"
      echo "  - branch: master"
      echo "Default run: $0 --model Qwen3.5-35B-A3B-p32768d1024 --op mm --cache-dir /root/.flaggems --dtypes bfloat16 --warmup 100 --parallel 8"
      echo "Example 1: $0 --model Qwen3.5-35B-A3B-p32768d1024 --op mm"
      echo "Example 2: $0 --yaml Qwen3.5-35B-A3B-p32768d1024 --op mm"
      echo "Example 3: $0 --yaml Qwen3.5-35B-A3B-p32768d1024 --branch --op w8a8_block_fp8_matmul"
      echo "Example 4: $0 --model Qwen3.5-35B-A3B-p32768d1024 --clear-cache --op mm"
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
  local use_flagtune="${4:-false}"
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

  if [[ "${use_flagtune,,}" == "true" ]]; then
    run_cmd env USE_FLAGTUNE=1 pytest "${pytest_args[@]}"
  else
    run_cmd pytest "${pytest_args[@]}"
  fi
}

REPORT_SUFFIX=""
if [[ "${BRANCH_COMPARE,,}" == "true" ]]; then
  REPORT_SUFFIX="_$(echo "$COMPARE_BRANCH" | tr '/ ' '__')"
fi
REPORT_MD="$FLAGTUNE_DIR/reports/${RUN_NAME}_${OP}${REPORT_SUFFIX}.md"
REPORT_XLSX="$FLAGTUNE_DIR/reports/${RUN_NAME}_${OP}${REPORT_SUFFIX}.xlsx"
REPORT_GENERATED=false
ORIGINAL_GIT_REF=""

checkout_git_ref() {
  local target_ref="$1"

  if ! git rev-parse --verify --quiet "${target_ref}^{commit}" >/dev/null; then
    echo "[ERROR] Git ref not found: ${target_ref}"
    exit 1
  fi

  if ! run_cmd git checkout "$target_ref"; then
    echo "[ERROR] Failed to checkout ${target_ref}."
    exit 1
  fi
}

get_current_branch_label() {
  local branch_name
  branch_name="$(git branch --show-current)"
  if [[ -n "$branch_name" ]]; then
    echo "$branch_name"
  else
    echo "HEAD-$(git rev-parse --short HEAD)"
  fi
}

get_current_git_ref() {
  local branch_name
  branch_name="$(git branch --show-current)"
  if [[ -n "$branch_name" ]]; then
    echo "$branch_name"
  else
    git rev-parse HEAD
  fi
}

restore_original_git_ref() {
  local exit_code="$1"
  local current_ref=""

  trap - EXIT

  if [[ -n "$ORIGINAL_GIT_REF" ]]; then
    current_ref="$(get_current_git_ref)"
    if [[ "$current_ref" != "$ORIGINAL_GIT_REF" ]]; then
      echo "[INFO] Switching back to ${ORIGINAL_GIT_REF}."
      if ! git checkout "$ORIGINAL_GIT_REF"; then
        echo "[ERROR] Failed to checkout ${ORIGINAL_GIT_REF}."
        if [[ "$exit_code" -eq 0 ]]; then
          exit 1
        fi
      fi
    fi
  fi

  exit "$exit_code"
}

trap 'restore_original_git_ref "$?"' EXIT

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

if [[ "${BRANCH_COMPARE,,}" == "true" ]]; then
  CURRENT_BRANCH_LABEL="$(get_current_branch_label)"
  ORIGINAL_GIT_REF="$(get_current_git_ref)"
  echo "[INFO] Using --branch mode: comparing ${COMPARE_BRANCH} vs ${CURRENT_BRANCH_LABEL}."
  checkout_git_ref "$COMPARE_BRANCH"
  run_mm_benchmark "$COMPARE_BRANCH" "$SHAPE_FILE" "$CLEAR_CACHE" false
  if [[ "$ORIGINAL_GIT_REF" != "$COMPARE_BRANCH" ]]; then
    checkout_git_ref "$ORIGINAL_GIT_REF"
  fi
  run_mm_benchmark "$CURRENT_BRANCH_LABEL" "$SHAPE_FILE" "$CLEAR_CACHE" false

  echo "[INFO] Generating pretune report."
  run_cmd python3 "$FLAGTUNE_DIR/processing/summary.py" \
    --model "$RUN_NAME" \
    --op "$OP" \
    --output-suffix "$REPORT_SUFFIX" \
    --left-stage-label "$COMPARE_BRANCH" \
    --right-stage-label "$CURRENT_BRANCH_LABEL" \
    --left-report-label "$COMPARE_BRANCH" \
    --right-report-label "$CURRENT_BRANCH_LABEL"
else
  run_mm_benchmark "default" "$SHAPE_FILE" "$CLEAR_CACHE" false
  run_mm_benchmark "expand" "$SHAPE_FILE" "$CLEAR_CACHE" true

  echo "[INFO] Generating pretune report."
  run_cmd python3 "$FLAGTUNE_DIR/processing/summary.py" --model "$RUN_NAME" --op "$OP" --output-suffix "$REPORT_SUFFIX"
fi

REPORT_GENERATED=true


echo

echo "[DONE] Flagtune finished successfully."
echo "[LOG] Output Log saved to $LOG_FILE."
if [[ "${REPORT_GENERATED,,}" == "true" ]]; then
  echo "[REPORT] Markdown report saved to $REPORT_MD."
  echo "[REPORT] Excel report saved to $REPORT_XLSX."
fi
