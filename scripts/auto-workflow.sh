#!/usr/bin/env bash
#
# auto-workflow.sh - FlagTune 一键基准测试脚本
#
# 用法:
#   ./auto-workflow.sh --model qwen-3.5 --optimized
#   ./auto-workflow.sh --model qwen-3.5 --nsys
#   ./auto-workflow.sh --model qwen-3.5 --shape
#   ./auto-workflow.sh --model qwen-3.5 --all
#
# 参数:
#   --model NAME          使用 config.yaml.NAME 作为配置文件
#   --optimized           运行优化模式 (cuda + gems)
#   --nsys                运行 nsys profiling 模式 (cuda + gems)
#   --shape               运行 shape 分析模式 (gems with GEMS_ONCE=false)
#   --all                 依次运行 shape → optimized → nsys
#   --device N            GPU 设备 ID (默认 0)
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
DEVICE=0
MODE=""

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL_CONFIG="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
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

# 构建 run-workflow 命令的基础参数
build_base_args() {
    local args=()
    args+=("--device" "$DEVICE")
    args+=("--model" "$MODEL_CONFIG")
    echo "${args[@]}"
}

# 运行 optimized 模式
run_optimized() {
    log_section "运行 Optimized 模式"

    local base_args
    base_args=$(build_base_args)

    log_step "1/2: CUDA 优化模式"
    "${SCRIPT_DIR}/run-workflow.sh" --mode cuda $base_args --optimized

    log_step "2/2: GEMS 优化模式"
    "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode all $base_args --optimized

    log_info "Optimized 模式完成"
}

# 运行 nsys 模式
run_nsys() {
    log_section "运行 NSYS Profiling 模式"

    local base_args
    base_args=$(build_base_args)

    log_step "1/2: CUDA NSYS Profiling"
    "${SCRIPT_DIR}/run-workflow.sh" --mode cuda $base_args --optimized --nsys

    log_step "2/2: GEMS NSYS Profiling"
    "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode all $base_args --optimized --nsys

    log_info "NSYS Profiling 模式完成"
}

# 运行 shape 模式
run_shape() {
    log_section "运行 Shape 分析模式"

    local base_args
    base_args=$(build_base_args)

    log_step "GEMS Shape 分析 (GEMS_ONCE=false)"
    "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode all $base_args --optimized --gems-once false

    log_info "Shape 模式完成"
}

# 运行所有模式
run_all() {
    log_section "运行所有模式 (shape → optimized → nsys)"

    run_shape
    echo ""

    run_optimized
    echo ""

    run_nsys

    log_info "所有模式完成"
}

# 主函数
main() {
    parse_args "$@"
    validate_args

    log_info "FlagTune 一键基准测试"
    log_info "模型: $MODEL_CONFIG"
    log_info "设备: $DEVICE"
    log_info "模式: $MODE"

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
