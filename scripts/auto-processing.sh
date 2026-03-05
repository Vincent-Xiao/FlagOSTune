#!/usr/bin/env bash
#
# auto-processing.sh - FlagTune 一键数据处理脚本
#
# 用法:
#   ./auto-processing.sh --model qwen-3.5 --optimized
#   ./auto-processing.sh --model qwen-3.5 --nsys
#   ./auto-processing.sh --model qwen-3.5 --nsys --skip-export
#   ./auto-processing.sh --model qwen-3.5 --shape
#   ./auto-processing.sh --model qwen-3.5 --all
#
# 参数:
#   --model NAME          使用 config.yaml.NAME 作为配置文件
#   --optimized           运行基准测试统计 (bench -f optimized)
#   --nsys                运行 Nsys 性能分析
#   --skip-export         与 --nsys 配合使用，跳过 nsys 导出步骤
#   --shape               运行 Shape 分析
#   --all                 依次运行 shape → optimized → nsys
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
MODE=""
SKIP_EXPORT=false

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL_CONFIG="$2"
                shift 2
                ;;
            --optimized)
                MODE="optimized"
                shift
                ;;
            --nsys)
                MODE="nsys"
                shift
                ;;
            --skip-export)
                SKIP_EXPORT=true
                shift
                ;;
            --shape)
                MODE="shape"
                shift
                ;;
            --all)
                MODE="all"
                shift
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

    if [[ -z "$MODE" ]]; then
        log_error "必须指定运行模式: --optimized, --nsys, --shape 或 --all"
        exit 1
    fi
}

# 构建基础参数
build_base_args() {
    local args=()
    args+=("--model" "$MODEL_CONFIG")
    echo "${args[@]}"
}

# 运行 optimized 模式 (基准测试统计)
run_optimized() {
    log_section "运行基准测试统计 (Optimized)"

    local base_args
    base_args=$(build_base_args)

    log_step "处理优化模式基准测试结果"
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

# 运行 shape 模式
run_shape() {
    log_section "运行 Shape 分析"

    local base_args
    base_args=$(build_base_args)

    log_step "分析 Gems Shape 信息"
    "${SCRIPT_DIR}/run-processing.sh" --workflow shape $base_args

    log_info "Shape 分析完成"
}

# 运行所有模式
run_all() {
    log_section "运行所有处理流程 (shape → optimized → nsys)"

    run_shape
    echo ""

    run_optimized
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
    log_info "模式: $MODE"
    if [[ "$MODE" == "nsys" && "$SKIP_EXPORT" == true ]]; then
        log_info "跳过导出: 是"
    fi

    cd "$PROJECT_ROOT"

    case "$MODE" in
        optimized)
            run_optimized
            ;;
        nsys)
            run_nsys
            ;;
        shape)
            run_shape
            ;;
        all)
            run_all
            ;;
    esac

    log_info "完成!"
}

main "$@"
