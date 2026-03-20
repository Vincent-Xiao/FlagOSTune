#!/usr/bin/env bash
#
# auto-workflow.sh - FlagTune 一键基准测试脚本
#
# 用法:
#   ./auto-workflow.sh --model qwen-3.5
#   ./auto-workflow.sh --model qwen-3.5 --scenario optimized
#   ./auto-workflow.sh --model qwen-3.5 --scenario full
#   ./auto-workflow.sh --model qwen-3.5 --mode cuda --scenario optimized
#   ./auto-workflow.sh --model qwen-3.5 --mode gems --scenario optimized
#   ./auto-workflow.sh --model qwen-3.5 --scenario shape
#   ./auto-workflow.sh --model qwen-3.5 --nsys
#   ./auto-workflow.sh --model qwen-3.5 --mode cuda --nsys
#   ./auto-workflow.sh --model qwen-3.5 --mode gems --nsys
#   ./auto-workflow.sh --model qwen-3.5 --torch
#   ./auto-workflow.sh --model qwen-3.5 --scenario shape --gems-mode mm
#   ./auto-workflow.sh --model qwen-3.5 --scenario shape --gems-once true
#   ./auto-workflow.sh --model qwen-3.5 --scenario shape --gems-mode mm --pretune
#   ./auto-workflow.sh --model qwen-3.5 --all
#
# 参数:
#   --model NAME          使用 config.yaml.NAME 作为配置文件
#   --mode TYPE           运行目标 (cuda|gems|all，默认 all)
#   --scenario TYPE       场景类型 (optimized|full|shape，默认 optimized)
#   --nsys                运行 nsys profiling 模式 (cuda + gems)
#   --torch               运行 torch profiling 模式 (cuda + gems)
#   --gems-mode MODE      FlagGems 模式 (默认 all)
#   --gems-once BOOL      透传给 run-workflow.sh 的 GEMS_ONCE (true|false，默认 true)
#   --pretune             透传给 run-workflow.sh，输出目录追加 pretune 后缀
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
WORKFLOW_MODE="scenario"
TARGET_MODE="all"
SCENARIO="optimized"
GEMS_MODE="all"
GEMS_ONCE="true"
PRETUNE=false

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
            --mode)
                TARGET_MODE="$2"
                shift 2
                ;;
            --scenario|--scnario)
                SCENARIO="$2"
                shift 2
                ;;
            --nsys)
                WORKFLOW_MODE="nsys"
                shift
                ;;
            --torch)
                WORKFLOW_MODE="torch"
                shift
                ;;
            --gems-mode)
                GEMS_MODE="$2"
                shift 2
                ;;
            --gems-once)
                GEMS_ONCE="$2"
                shift 2
                ;;
            --pretune)
                PRETUNE=true
                shift
                ;;
            --all)
                WORKFLOW_MODE="all"
                shift
                ;;
            -h|--help)
                head -33 "$0" | tail -31
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

    case "$TARGET_MODE" in
        cuda|gems|all)
            ;;
        *)
            log_error "--mode 仅支持: cuda, gems, all；当前值: $TARGET_MODE"
            exit 1
            ;;
    esac

    if [[ "$GEMS_ONCE" != "true" && "$GEMS_ONCE" != "false" ]]; then
        log_error "--gems-once 仅支持: true, false；当前值: $GEMS_ONCE"
        exit 1
    fi
}

# 构建 run-workflow 命令的基础参数
build_base_args() {
    local args=()
    args+=("--device" "$DEVICE")
    args+=("--model" "$MODEL_CONFIG")
    if [[ "$PRETUNE" == "true" ]]; then
        args+=("--pretune")
    fi
    echo "${args[@]}"
}

run_by_target_mode() {
    local scenario="$1"
    local extra_flag="${2:-}"
    local base_args
    local gems_once_arg=""
    base_args=$(build_base_args)

    gems_once_arg="--gems-once $GEMS_ONCE"

    case "$TARGET_MODE" in
        cuda)
            if [[ "$scenario" == "shape" ]]; then
                log_error "--mode cuda 不支持 shape 场景；请使用 --mode gems 或 --mode all"
                exit 1
            fi
            log_step "CUDA ${extra_flag#-- } ${scenario}"
            "${SCRIPT_DIR}/run-workflow.sh" --mode cuda $base_args --scenario "$scenario" ${extra_flag:+$extra_flag}
            ;;
        gems)
            if [[ -n "$extra_flag" ]]; then
                log_step "GEMS ${extra_flag#-- } ${scenario}"
            elif [[ "$scenario" == "shape" ]]; then
                log_step "GEMS Shape 场景"
            else
                log_step "GEMS 场景 (${scenario})"
            fi
            if [[ "$scenario" == "shape" ]]; then
                "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode "$GEMS_MODE" $base_args --scenario shape $gems_once_arg ${extra_flag:+$extra_flag}
            else
                "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode "$GEMS_MODE" $base_args --scenario "$scenario" $gems_once_arg ${extra_flag:+$extra_flag}
            fi
            ;;
        all)
            if [[ "$scenario" == "shape" ]]; then
                log_info "shape 场景仅运行 GEMS"
                log_step "GEMS Shape 场景"
                "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode "$GEMS_MODE" $base_args --scenario shape $gems_once_arg ${extra_flag:+$extra_flag}
            else
                if [[ -n "$extra_flag" ]]; then
                    log_step "CUDA ${extra_flag#-- } ${scenario}"
                else
                    log_step "CUDA 场景 (${scenario})"
                fi
                "${SCRIPT_DIR}/run-workflow.sh" --mode cuda $base_args --scenario "$scenario" ${extra_flag:+$extra_flag}

                if [[ -n "$extra_flag" ]]; then
                    log_step "GEMS ${extra_flag#-- } ${scenario}"
                else
                    log_step "GEMS 场景 (${scenario})"
                fi
                "${SCRIPT_DIR}/run-workflow.sh" --mode gems --gems-mode "$GEMS_MODE" $base_args --scenario "$scenario" $gems_once_arg ${extra_flag:+$extra_flag}
            fi
            ;;
    esac
}

# 运行指定场景
run_scenario() {
    if [[ "$SCENARIO" == "shape" ]]; then
        log_section "运行 Shape 场景 (GEMS_ONCE=${GEMS_ONCE})"
    else
        log_section "运行 ${SCENARIO} 场景"
    fi

    run_by_target_mode "$SCENARIO"
    log_info "场景运行完成"
}

# 运行 nsys 模式
run_nsys() {
    log_section "运行 NSYS Profiling 模式"

    local profile_scenario="optimized"
    if [[ "$SCENARIO" != "$profile_scenario" ]]; then
        log_warn "--nsys 模式固定使用 optimized 场景，忽略 --scenario=$SCENARIO"
    fi

    run_by_target_mode "$profile_scenario" "--nsys"
    log_info "NSYS Profiling 模式完成"
}

# 运行 torch 模式
run_torch() {
    log_section "运行 Torch Profiling 模式"

    local profile_scenario="optimized"
    if [[ "$SCENARIO" != "$profile_scenario" ]]; then
        log_warn "--torch 模式固定使用 optimized 场景，忽略 --scenario=$SCENARIO"
    fi

    run_by_target_mode "$profile_scenario" "--torch"
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
    log_info "工作流: $WORKFLOW_MODE"
    log_info "运行目标: $TARGET_MODE"
    log_info "场景: $SCENARIO"
    log_info "GEMS 模式: $GEMS_MODE"
    log_info "GEMS_ONCE: ${GEMS_ONCE}"
    log_info "Pretune: $PRETUNE"

    cd "$PROJECT_ROOT"

    case "$WORKFLOW_MODE" in
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
