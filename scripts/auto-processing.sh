#!/usr/bin/env bash
#
# auto-processing.sh - FlagTune 一键数据处理脚本
#
# 用法:
#   ./auto-processing.sh --model qwen-3.5 --workflow bench
#   ./auto-processing.sh --model qwen-3.5 --workflow shape
#   ./auto-processing.sh --model qwen-3.5 --workflow shape --gems-mode mm
#   ./auto-processing.sh --model qwen-3.5 --workflow nsys
#   ./auto-processing.sh --model qwen-3.5 --workflow nsys --skip-export
#   ./auto-processing.sh --model qwen-3.5 --workflow torch
#   ./auto-processing.sh --model qwen-3.5 --workflow torch --mode cuda
#   ./auto-processing.sh --model qwen-3.5 --warmup 2
#   ./auto-processing.sh --model qwen-3.5 --workflow all
#
# 参数:
#   --model NAME          使用 config.yaml.NAME 作为配置文件
#   --workflow TYPE       工作流类型 (bench|nsys|torch|shape|all，默认 bench)
#   --mode TYPE           Torch 工作流模式 (cuda|gems|compare，默认 compare)
#   --rank VALUE          Torch 工作流 rank 选择（数字或 all，默认 0）
#   --gems-mode MODE      Shape 工作流使用的 FlagGems 模式 (默认 all)
#   --warmup N            跳过预热轮数 (默认 2，bench/optimized 生效)
#   --skip-export         与 --workflow nsys 配合使用，跳过 nsys 导出步骤
#   --report              透传给 run-processing.sh，仅执行报告生成相关逻辑
#

set -euo pipefail

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 颜色输出
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

# 默认参数
MODEL_CONFIG=""
WORKFLOW="bench"
SKIP_EXPORT=false
REPORT=false
WARMUP=2
GEMS_MODE="all"
TORCH_MODE="compare"
TORCH_RANK="0"

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL_CONFIG="$2"
                shift 2
                ;;
            --workflow)
                WORKFLOW="$2"
                shift 2
                ;;
            --mode)
                TORCH_MODE="$2"
                shift 2
                ;;
            --rank)
                TORCH_RANK="$2"
                shift 2
                ;;
            --gems-mode)
                GEMS_MODE="$2"
                shift 2
                ;;
            --skip-export)
                SKIP_EXPORT=true
                shift
                ;;
            --report)
                REPORT=true
                shift
                ;;
            --warmup)
                WARMUP="$2"
                shift 2
                ;;
            -h|--help)
                head -25 "$0" | tail -23
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
}

# 验证参数
validate_args() {
    if [[ -z "$MODEL_CONFIG" ]]; then
        log_error "必须指定 --model 参数"
        exit 1
    fi

    case "$WORKFLOW" in
        bench|nsys|torch|shape|all)
            ;;
        *)
            log_error "--workflow 仅支持: bench, nsys, torch, shape, all；当前值: $WORKFLOW"
            exit 1
            ;;
    esac

    case "$TORCH_MODE" in
        cuda|gems|compare)
            ;;
        *)
            log_error "--mode 仅支持: cuda, gems, compare；当前值: $TORCH_MODE"
            exit 1
            ;;
    esac

    if [[ ! "$TORCH_RANK" =~ ^[0-9]+$ && "$TORCH_RANK" != "all" ]]; then
        log_error "--rank 仅支持数字或 all；当前值: $TORCH_RANK"
        exit 1
    fi
}

# 构建基础参数
build_base_args() {
    local args=()
    args+=("--model" "$MODEL_CONFIG")
    args+=("--warmup" "$WARMUP")
    if [[ "$REPORT" == true ]]; then
        args+=("--report")
    fi
    args+=("--gems-mode" "$GEMS_MODE")
    args+=("--mode" "$TORCH_MODE")
    args+=("--rank" "$TORCH_RANK")
    echo "${args[@]}"
}

# 运行 optimized 场景基准测试统计
run_bench() {
    log_section "运行基准测试统计 (bench)"

    local base_args
    base_args=$(build_base_args)

    log_step "处理基准测试结果"
    "${SCRIPT_DIR}/run-processing.sh" --workflow bench -f optimized $base_args

    log_info "基准测试统计完成"
}

# 运行 nsys 模式
run_nsys() {
    log_section "运行 Nsys 性能分析"

    local base_args
    base_args=$(build_base_args)

    local skip_args=""
    if [[ "$SKIP_EXPORT" == true ]]; then
        skip_args="--skip-export"
    fi

    log_step "分析 Nsys 性能数据"
    "${SCRIPT_DIR}/run-processing.sh" --workflow nsys $base_args $skip_args

    log_info "Nsys 分析完成"
}

# 运行 shape 场景
run_shape() {
    log_section "运行 Shape 分析"

    local base_args
    base_args=$(build_base_args)

    log_step "分析 Gems Shape 信息"
    "${SCRIPT_DIR}/run-processing.sh" --workflow shape $base_args

    log_info "Shape 分析完成"
}

# 运行 torch profiler 场景
run_torch() {
    log_section "运行 Torch profiler 性能分析"

    local base_args
    base_args=$(build_base_args)

    log_step "分析 Torch profiler 性能数据"
    "${SCRIPT_DIR}/run-processing.sh" --workflow torch $base_args

    log_info "Torch profiler 分析完成"
}

# 运行所有模式
run_all() {
    log_section "运行所有处理流程 (shape → optimized → nsys)"

    if [[ "$REPORT" == true ]]; then
        local base_args
        base_args=$(build_base_args)

        log_step "仅执行报告生成相关流程"
        "${SCRIPT_DIR}/run-processing.sh" --workflow all $base_args
        log_info "报告生成完成"
        return 0
    fi

    run_shape
    echo ""

    run_bench
    echo ""

    run_nsys

    log_info "所有处理流程完成"
}

# 主函数
main() {
    parse_args "$@"
    validate_args

    log_info "FlagTune 一键数据处理"
    log_info "模型: $MODEL_CONFIG"
    log_info "工作流: $WORKFLOW"
    log_info "Torch Mode: $TORCH_MODE"
    log_info "Torch Rank: $TORCH_RANK"
    log_info "Gems Mode: $GEMS_MODE"
    log_info "Warmup: $WARMUP"
    if [[ "$REPORT" == true ]]; then
        log_info "生成报告: 是 (--report)"
    fi
    if [[ "$WORKFLOW" == "nsys" && "$SKIP_EXPORT" == true ]]; then
        log_info "跳过导出: 是"
    fi

    cd "$PROJECT_ROOT"

    case "$WORKFLOW" in
        bench)
            run_bench
            ;;
        shape)
            run_shape
            ;;
        nsys)
            run_nsys
            ;;
        torch)
            run_torch
            ;;
        all)
            run_all
            ;;
    esac

    log_info "完成!"
}

main "$@"
