#!/usr/bin/env bash
#
# auto-workflow.sh - FlagTune 一键基准测试脚本
#
# 用法:
#   ./auto-workflow.sh --model qwen-3.5
#   ./auto-workflow.sh --model qwen-3.5 --scenario optimized
#   ./auto-workflow.sh --model qwen-3.5 --scenario full
#   ./auto-workflow.sh --model qwen-3.5 --scenario shape
#   ./auto-workflow.sh --model qwen-3.5 --nsys
#   ./auto-workflow.sh --model qwen-3.5 --nsys --cuda
#   ./auto-workflow.sh --model qwen-3.5 --nsys --gems
#   ./auto-workflow.sh --model qwen-3.5 --torch
#   ./auto-workflow.sh --model qwen-3.5 --all
#
# 参数:
#   --model NAME          使用 config.yaml.NAME 作为配置文件
#   --scenario TYPE       场景类型 (optimized|full|shape，默认 optimized)
#   --nsys                运行 nsys profiling 模式 (cuda + gems)
#   --torch               运行 torch profiling 模式 (cuda + gems)
#   --cuda                与 --nsys/--torch 配合使用，仅运行 CUDA profiling
#   --gems                与 --nsys/--torch 配合使用，仅运行 GEMS profiling
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
MODE="scenario"
SCENARIO="optimized"
NSYS_TARGET=""  # cuda, gems, or empty (both)

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
            --scenario|--scnario)
                SCENARIO="$2"
                shift 2
                ;;
            --nsys)
                MODE="nsys"
                shift
                ;;
            --torch)
                MODE="torch"
                shift
                ;;
            --cuda)
                NSYS_TARGET="cuda"
                shift
                ;;
            --gems)
                NSYS_TARGET="gems"
                shift
                ;;
            --all)
                MODE="all"
                shift
                ;;
            -h|--help)
                head -28 "$0" | tail -26
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

    case "$SCENARIO" in
        optimized|full|shape)
            ;;
        *)
            log_error "--scenario 仅支持: optimized, full, shape；当前值: $SCENARIO"
            exit 1
            ;;
    esac
}

# 构建 run-workflow 命令的基础参数
build_base_args() {
    local args=()
    args+=("--device" "$DEVICE")
    args+=("--model" "$MODEL_CONFIG")
    echo "${args[@]}"
}

# 运行指定场景
run_scenario() {
    if [[ "$SCENARIO" == "shape" ]]; then
        log_section "运行 Shape 场景 (GEMS_ONCE=false)"
    else
        log_section "运行 ${SCENARIO} 场景"
    fi

    local base_args
    base_args=$(build_base_args)

    if [[ "$SCENARIO" == "shape" ]]; then
        log_step "1/1: GEMS Shape 场景"
        "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode all $base_args --scenario shape --gems-once false
        log_info "Shape 场景完成"
        return 0
    fi

    log_step "1/2: CUDA 场景 (${SCENARIO})"
    "${SCRIPT_DIR}/run-workflow.sh" --mode cuda $base_args --scenario "$SCENARIO"

    log_step "2/2: GEMS 场景 (${SCENARIO})"
    "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode all $base_args --scenario "$SCENARIO"

    log_info "场景运行完成"
}

# 运行 nsys 模式
run_nsys() {
    log_section "运行 NSYS Profiling 模式"

    local profile_scenario="optimized"
    if [[ "$SCENARIO" != "$profile_scenario" ]]; then
        log_warn "--nsys 模式固定使用 optimized 场景，忽略 --scenario=$SCENARIO"
    fi

    local base_args
    base_args=$(build_base_args)

    # 根据 NSYS_TARGET 决定运行哪些
    if [[ -z "$NSYS_TARGET" || "$NSYS_TARGET" == "cuda" ]]; then
        log_step "CUDA NSYS Profiling (${profile_scenario})"
        "${SCRIPT_DIR}/run-workflow.sh" --mode cuda $base_args --scenario "$profile_scenario" --nsys
    fi

    if [[ -z "$NSYS_TARGET" || "$NSYS_TARGET" == "gems" ]]; then
        log_step "GEMS NSYS Profiling (${profile_scenario})"
        "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode all $base_args --scenario "$profile_scenario" --nsys
    fi

    log_info "NSYS Profiling 模式完成"
}

# 运行 torch 模式
run_torch() {
    log_section "运行 Torch Profiling 模式"

    local profile_scenario="optimized"
    if [[ "$SCENARIO" != "$profile_scenario" ]]; then
        log_warn "--torch 模式固定使用 optimized 场景，忽略 --scenario=$SCENARIO"
    fi

    local base_args
    base_args=$(build_base_args)

    # 复用 --cuda/--gems 目标筛选
    if [[ -z "$NSYS_TARGET" || "$NSYS_TARGET" == "cuda" ]]; then
        log_step "CUDA Torch Profiling (${profile_scenario})"
        "${SCRIPT_DIR}/run-workflow.sh" --mode cuda $base_args --scenario "$profile_scenario" --torch
    fi

    if [[ -z "$NSYS_TARGET" || "$NSYS_TARGET" == "gems" ]]; then
        log_step "GEMS Torch Profiling (${profile_scenario})"
        "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode all $base_args --scenario "$profile_scenario" --torch
    fi

    log_info "Torch Profiling 模式完成"
}

# 运行所有模式
run_all() {
    log_section "运行所有模式 (shape → optimized → nsys)"

    local original_scenario="$SCENARIO"

    SCENARIO="shape"
    run_scenario
    echo ""

    SCENARIO="optimized"
    run_scenario
    echo ""

    SCENARIO="optimized"
    run_nsys

    SCENARIO="$original_scenario"

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
    log_info "场景: $SCENARIO"
    if [[ "$MODE" == "nsys" && -n "$NSYS_TARGET" ]]; then
        log_info "NSYS 目标: $NSYS_TARGET"
    fi
    if [[ "$MODE" == "torch" && -n "$NSYS_TARGET" ]]; then
        log_info "Torch 目标: $NSYS_TARGET"
    fi

    cd "$PROJECT_ROOT"

    case "$MODE" in
        scenario)
            run_scenario
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
