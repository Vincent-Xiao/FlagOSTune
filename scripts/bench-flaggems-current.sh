#!/usr/bin/env bash
#
# bench-flaggems-current.sh - Benchmark vLLM patches one by one on native CUDA.
#
# For each operator this applies a single patch-vllm patch, runs the benchmark
# in --mode cuda (no FlagGems global dispatch), then restores the patch. This
# isolates the end-to-end impact of one hand-applied operator patch.
#
# Usage:
#   ./scripts/bench-flaggems-current.sh
#   ./scripts/bench-flaggems-current.sh --only w8a8,mm
#   ./scripts/bench-flaggems-current.sh --only fused-marlin-moe
#   ./scripts/bench-flaggems-current.sh --dry-run
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_section() { echo -e "\n${CYAN}========================================${NC}"; echo -e "${CYAN}$1${NC}"; echo -e "${CYAN}========================================${NC}\n"; }

MODEL_CONFIG="DeepSeek-V4-Flash"
DEVICE="0"
RUNS="5"
ONLY_LIST=""
DRY_RUN=false
CONTINUE_ON_ERROR=false
SKIP_CUDA_BASELINE=false
REPORT_DATE="$(date +%F)"
ARCHIVE_RUN_ID="$(date +%Y%m%d-%H%M%S)"
CURRENT_OP=""
CURRENT_APPLIED=false

PATCH_OPS=(
    "w8a8"
    "router-gemm"
    "fused-marlin-moe"
    "compute-global-topk-indices-and-lens"
    "topk-softplus-sqrt"
    "flashmla-sparse"
    "flashmla-with-kvcache"
    "fp8-einsum"
    "indexer-k-quant"
    "cp-gather-indexer"
    "per-token-group-fp8"
    "mm"
)

canonical_op_name() {
    local op="$1"
    op="${op// /}"
    op="${op//_/-}"

    case "$op" in
        w8a8|w8a8-block-fp8-matmul)
            printf '%s\n' "w8a8"
            ;;
        mm)
            printf '%s\n' "mm"
            ;;
        router-gemm|router-gemm-bf16-fp32)
            printf '%s\n' "router-gemm"
            ;;
        fused-marlin-moe)
            printf '%s\n' "fused-marlin-moe"
            ;;
        flashmla-sparse|flash-mla-sparse)
            printf '%s\n' "flashmla-sparse"
            ;;
        flashmla-with-kvcache|flash-mla-with-kvcache)
            printf '%s\n' "flashmla-with-kvcache"
            ;;
        fp8-einsum|deepseek-v4-fp8-einsum)
            printf '%s\n' "fp8-einsum"
            ;;
        indexer-k-quant|indexer-k-quant-and-cache)
            printf '%s\n' "indexer-k-quant"
            ;;
        cp-gather-indexer|cp-gather-indexer-k-quant-cache)
            printf '%s\n' "cp-gather-indexer"
            ;;
        per-token-group-fp8|per-token-group-fp8-quant)
            printf '%s\n' "per-token-group-fp8"
            ;;
        compute-global-topk-indices-and-lens)
            printf '%s\n' "compute-global-topk-indices-and-lens"
            ;;
        topk-softplus-sqrt)
            printf '%s\n' "topk-softplus-sqrt"
            ;;
        marlin-moe)
            printf '%s\n' "$op"
            ;;
        *)
            printf '%s\n' "$op"
            ;;
    esac
}

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --model NAME             Model config name (default: DeepSeek-V4-Flash)
  --device N               GPU device id (default: 0)
  --runs N                 Benchmark runs (default: 5)
  --only op1,op2           Run only selected operators from the list below
  --skip-cuda-baseline     Skip the initial clean CUDA baseline run
  --dry-run                Print commands and report moves without running them
  --continue-on-error      Restore failed operator and continue with the next one
  -h, --help               Show this help

Default operators:
  ${PATCH_OPS[*]}
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                MODEL_CONFIG="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --runs)
                RUNS="$2"
                shift 2
                ;;
            --only)
                ONLY_LIST="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-cuda-baseline)
                SKIP_CUDA_BASELINE=true
                shift
                ;;
            --continue-on-error)
                CONTINUE_ON_ERROR=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                usage
                exit 1
                ;;
        esac
    done
}

validate_args() {
    if [[ -z "$MODEL_CONFIG" ]]; then
        log_error "--model must not be empty"
        exit 1
    fi
    if [[ ! "$DEVICE" =~ ^[0-9]+$ ]]; then
        log_error "--device must be a non-negative integer: $DEVICE"
        exit 1
    fi
    if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
        log_error "--runs must be a positive integer: $RUNS"
        exit 1
    fi
}

should_run_op() {
    local op="$1"
    local part

    [[ "$op" == "marlin-moe" ]] && return 1
    [[ -z "$ONLY_LIST" ]] && return 0

    IFS=',' read -ra parts <<< "$ONLY_LIST"
    for part in "${parts[@]}"; do
        part="$(canonical_op_name "$part")"
        if [[ "$part" == "$op" ]]; then
            return 0
        fi
    done
    return 1
}

validate_only_list() {
    local part op found

    [[ -z "$ONLY_LIST" ]] && return 0

    IFS=',' read -ra parts <<< "$ONLY_LIST"
    for part in "${parts[@]}"; do
        part="$(canonical_op_name "$part")"
        [[ -z "$part" ]] && continue

        if [[ "$part" == "marlin-moe" ]]; then
            log_warn "Skipping marlin-moe: this script intentionally excludes it"
            continue
        fi

        found=false
        for op in "${PATCH_OPS[@]}"; do
            if [[ "$op" == "$part" ]]; then
                found=true
                break
            fi
        done

        if [[ "$found" != true ]]; then
            log_error "Unknown operator in --only: $part"
            exit 1
        fi
    done
}

run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

restore_current_op() {
    if [[ -n "$CURRENT_OP" && "$CURRENT_APPLIED" == true ]]; then
        log_warn "Restoring patch for ${CURRENT_OP}"
        if [[ "$DRY_RUN" == true ]]; then
            printf '+'
            printf ' %q' "${SCRIPT_DIR}/patch-vllm-all.sh" --restore --only "$CURRENT_OP"
            printf '\n'
        else
            "${SCRIPT_DIR}/patch-vllm-all.sh" --restore --only "$CURRENT_OP" || true
        fi
        CURRENT_APPLIED=false
    fi
}

on_exit() {
    local status=$?
    if [[ $status -ne 0 ]]; then
        restore_current_op
    fi
    exit "$status"
}

report_paths_for_op() {
    local op="$1"
    local ext="$2"
    local report_dir="${PROJECT_ROOT}/reports/${MODEL_CONFIG}"
    local src="${report_dir}/bench-optimized-report-${REPORT_DATE}.${ext}"
    local dst="${report_dir}/bench-optimized-report-${REPORT_DATE}-cuda-patch-${op}.${ext}"
    printf '%s\n%s\n' "$src" "$dst"
}

check_report_targets_available() {
    local op="$1"
    local ext src dst

    for ext in md xlsx; do
        mapfile -t paths < <(report_paths_for_op "$op" "$ext")
        src="${paths[0]}"
        dst="${paths[1]}"

        if [[ "$DRY_RUN" != true && -e "$dst" ]]; then
            log_error "Target report already exists: $dst"
            return 1
        fi

        if [[ "$DRY_RUN" != true && ! -f "$src" ]]; then
            log_error "Expected report was not generated: $src"
            return 1
        fi
    done
}

rename_reports() {
    local op="$1"
    local ext src dst

    check_report_targets_available "$op"

    for ext in md xlsx; do
        mapfile -t paths < <(report_paths_for_op "$op" "$ext")
        src="${paths[0]}"
        dst="${paths[1]}"

        if [[ "$DRY_RUN" == true ]]; then
            printf '+ mv %q %q\n' "$src" "$dst"
        else
            mv "$src" "$dst"
            log_info "Saved report: $dst"
        fi
    done
}

archive_run_logs() {
    local label="$1"
    local source_dir="$2"
    local archive_root="${PROJECT_ROOT}/results/${MODEL_CONFIG}/bench_optimized_log/archive/${ARCHIVE_RUN_ID}"
    local target_dir="${archive_root}/${label}"
    local found=false
    local log_file

    if [[ "$DRY_RUN" == true ]]; then
        printf '+ mkdir -p %q\n' "$target_dir"
        printf '+ cp %q %q\n' "${source_dir}/*run*.log" "${target_dir}/"
        return 0
    fi

    if [[ ! -d "$source_dir" ]]; then
        log_warn "Run log source dir does not exist, skip archive: $source_dir"
        return 0
    fi

    mkdir -p "$target_dir"
    shopt -s nullglob
    for log_file in "${source_dir}"/*run*.log; do
        cp -p "$log_file" "$target_dir/"
        found=true
    done
    shopt -u nullglob

    if [[ "$found" == true ]]; then
        log_info "Archived run logs: $target_dir"
    else
        log_warn "No *run*.log found to archive under: $source_dir"
    fi
}

# 清空源日志目录中的 *run*.log，避免上一次运行的残留日志被误当成本次结果归档
clear_run_logs() {
    local source_dir="$1"

    if [[ "$DRY_RUN" == true ]]; then
        printf '+ rm -f %q\n' "${source_dir}/*run*.log"
        return 0
    fi

    [[ -d "$source_dir" ]] || return 0

    shopt -s nullglob
    local log_file removed=false
    for log_file in "${source_dir}"/*run*.log; do
        rm -f "$log_file"
        removed=true
    done
    shopt -u nullglob

    if [[ "$removed" == true ]]; then
        log_info "Cleared stale run logs: $source_dir"
    fi
}

run_cuda_baseline() {
    log_section "CUDA baseline"

    CURRENT_OP=""
    CURRENT_APPLIED=false

    local cuda_log_dir="${PROJECT_ROOT}/results/${MODEL_CONFIG}/bench_optimized_log/vllm_bench_cuda_logs"

    log_step "Restore all patches"
    run_cmd "${SCRIPT_DIR}/patch-vllm-all.sh" --restore || return $?

    log_step "Clear stale run logs"
    clear_run_logs "$cuda_log_dir"

    log_step "Run CUDA baseline benchmark"
    run_cmd "${SCRIPT_DIR}/auto-workflow.sh" \
        --model "$MODEL_CONFIG" \
        --device "$DEVICE" \
        --scenario optimized \
        --mode cuda \
        --runs "$RUNS" || return $?

    log_step "Process bench results"
    run_cmd "${SCRIPT_DIR}/auto-processing.sh" \
        --model "$MODEL_CONFIG" \
        --workflow bench || return $?

    log_step "Rename CUDA baseline reports"
    rename_reports "cuda" || return $?

    log_step "Archive CUDA run logs"
    archive_run_logs "cuda" "$cuda_log_dir" || return $?
}

run_one_op() {
    local op="$1"

    CURRENT_OP="$op"
    CURRENT_APPLIED=false

    local cuda_log_dir="${PROJECT_ROOT}/results/${MODEL_CONFIG}/bench_optimized_log/vllm_bench_cuda_logs"

    log_section "Benchmark patch: ${op}"

    log_step "Apply patch"
    run_cmd "${SCRIPT_DIR}/patch-vllm-all.sh" --apply --only "$op" || return $?
    CURRENT_APPLIED=true

    log_step "Clear stale run logs"
    clear_run_logs "$cuda_log_dir"

    log_step "Run CUDA benchmark (single patch: ${op})"
    run_cmd "${SCRIPT_DIR}/auto-workflow.sh" \
        --model "$MODEL_CONFIG" \
        --device "$DEVICE" \
        --mode cuda \
        --scenario optimized || return $?

    log_step "Process bench results"
    run_cmd "${SCRIPT_DIR}/auto-processing.sh" \
        --model "$MODEL_CONFIG" \
        --workflow bench || return $?

    log_step "Rename reports"
    rename_reports "$op" || return $?

    log_step "Archive CUDA run logs"
    archive_run_logs "$op" "$cuda_log_dir" || return $?

    log_step "Restore patch"
    run_cmd "${SCRIPT_DIR}/patch-vllm-all.sh" --restore --only "$op" || return $?
    CURRENT_APPLIED=false
    CURRENT_OP=""
}

main() {
    parse_args "$@"
    validate_args
    validate_only_list

    trap on_exit EXIT

    cd "$PROJECT_ROOT"

    log_info "Model: $MODEL_CONFIG"
    log_info "Device: $DEVICE"
    log_info "Runs: $RUNS"
    log_info "Report date: $REPORT_DATE"
    if [[ "$SKIP_CUDA_BASELINE" == true ]]; then
        log_warn "Skipping initial CUDA baseline"
    fi
    if [[ "$DRY_RUN" == true ]]; then
        log_warn "Dry run: commands will be printed but not executed"
    fi

    if [[ "$SKIP_CUDA_BASELINE" != true ]]; then
        run_cuda_baseline
    fi

    local op status failed=0 selected=0
    for op in "${PATCH_OPS[@]}"; do
        if ! should_run_op "$op"; then
            continue
        fi
        selected=$((selected + 1))

        set +e
        run_one_op "$op"
        status=$?
        set -e

        if [[ $status -ne 0 ]]; then
            failed=$((failed + 1))
            log_error "Operator failed: ${op}"
            restore_current_op
            if [[ "$CONTINUE_ON_ERROR" != true ]]; then
                exit "$status"
            fi
        fi
    done

    if [[ $selected -eq 0 ]]; then
        log_warn "No operators selected"
    fi

    if [[ $failed -ne 0 ]]; then
        log_error "Completed with ${failed} failed operator(s)"
        exit 1
    fi

    log_info "All selected operators completed"
}

main "$@"
