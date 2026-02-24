#!/usr/bin/env bash
#
# run-workflow.sh - FlagTune 通用工作流调度器
#
# 用法:
#   ./run-workflow.sh --mode cuda --device 0
#   ./run-workflow.sh --mode gems --device 0 --gems-mode all
#   ./run-workflow.sh --mode gems --device 0 --ops-file ops.txt
#
# 参数:
#   --mode cuda|gems      运行模式
#   --device N            GPU 设备 ID
#   --gems-mode MODE      FlagGems 模式 (all|NULL|算子名)
#   --ops-file FILE       算子列表文件
#   --optimized           优化模式
#   --nsys                Nsys 性能分析
#   --torch               Torch 性能分析
#   --run-idle            仅启动服务器不运行测试
#   --batch               批量模式 (逐算子运行)
#

set -euo pipefail

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL_CONFIG="${SCRIPT_DIR}/tools/tool_config.yaml"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

# 捕获当前 Python 解释器路径
Python_EXECUTABLE="${Python_EXECUTABLE:-$(which python3 2>/dev/null || echo 'python3')}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# 默认参数
MODE="cuda"
DEVICE=0
GEMS_MODE="all"
OPS_FILE=""
OPTIMIZED=false
NSYS_PROFILE=false
TORCH_PROFILE=false
RUN_IDLE=false
BATCH_MODE=false

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --gems-mode)
                GEMS_MODE="$2"
                shift 2
                ;;
            --ops-file)
                OPS_FILE="$2"
                shift 2
                ;;
            --optimized)
                OPTIMIZED=true
                shift
                ;;
            --nsys)
                NSYS_PROFILE=true
                shift
                ;;
            --torch)
                TORCH_PROFILE=true
                shift
                ;;
            --run-idle)
                RUN_IDLE=true
                shift
                ;;
            --batch)
                BATCH_MODE=true
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

# 检查依赖
check_dependencies() {
    local missing=()
    local required=(python3 tmux curl)

    for cmd in "${required[@]}"; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done

    # yq 是可选的，但强烈推荐
    if ! command -v yq &>/dev/null; then
        log_warn "yq 未安装，将使用内置简单 YAML 解析器"
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "缺少必需依赖: ${missing[*]}"
        log_info "请运行: ./scripts/setup-deps.sh"
        exit 1
    fi
}

# 读取配置
read_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi

    # 读取模型配置
    if command -v yq &>/dev/null; then
        MODEL_PATH=$(yq '.model.path' "$CONFIG_FILE")
        MODEL_NAME=$(yq '.model.name' "$CONFIG_FILE")
        TENSOR_PARALLEL=$(yq '.serve.tensor_parallel_size // .model.tensor_parallel_size' "$CONFIG_FILE")
        PORT_BASE=$(yq '.benchmark.port_base' "$CONFIG_FILE")
        NUM_RUNS=$(yq '.benchmark.num_runs' "$CONFIG_FILE")

        # 读取路径配置
        PATHS_RESULTS=$(yq '.paths.results // "results"' "$CONFIG_FILE")
        PATHS_REPORTS=$(yq '.paths.reports // "reports"' "$CONFIG_FILE")
        PATHS_USE_MODEL_NAME=$(yq '.paths.use_model_name // true' "$CONFIG_FILE")

        # 构建带模型名的路径前缀
        if [[ "$PATHS_USE_MODEL_NAME" == "true" ]]; then
            PATH_PREFIX="${PATHS_RESULTS}/${MODEL_NAME}"
            REPORT_PREFIX="${PATHS_REPORTS}/${MODEL_NAME}"
        else
            PATH_PREFIX="${PATHS_RESULTS}"
            REPORT_PREFIX="${PATHS_REPORTS}"
        fi

        # 读取场景配置
        if [[ "$OPTIMIZED" == "true" ]]; then
            SCENARIO_TYPE="optimized"
        else
            SCENARIO_TYPE="full"
        fi
    else
        log_error "需要 yq 来解析配置文件"
        exit 1
    fi

    PORT=$((PORT_BASE + DEVICE))
}

# 更新 tool_config.yaml
update_tool_config() {
    log_step "更新工具配置..."

    # 构建 gems_suffix
    local gems_suffix=""
    if [[ "$MODE" == "gems" ]]; then
        gems_suffix="_${GEMS_MODE}"
    fi

    # 构建其他后缀
    local optimized_suffix=""
    [[ "$OPTIMIZED" == "true" ]] && optimized_suffix="_optimized"

    local nsys_suffix=""
    [[ "$NSYS_PROFILE" == "true" ]] && nsys_suffix="_nsys_profile"

    local torch_suffix=""
    [[ "$TORCH_PROFILE" == "true" ]] && torch_suffix="_torch_profile"

    # 更新配置
    yq -i ".current_run.mode = \"$MODE\"" "$TOOL_CONFIG"
    yq -i ".current_run.device = $DEVICE" "$TOOL_CONFIG"
    yq -i ".current_run.port = $PORT" "$TOOL_CONFIG"
    yq -i ".current_run.gems.mode = \"$GEMS_MODE\"" "$TOOL_CONFIG"
    yq -i ".current_run.optimized = $OPTIMIZED" "$TOOL_CONFIG"
    yq -i ".current_run.nsys_profile = $NSYS_PROFILE" "$TOOL_CONFIG"
    yq -i ".current_run.torch_profile = $TORCH_PROFILE" "$TOOL_CONFIG"

    # 更新日志路径 (包含模型名)
    local log_dir="${PATH_PREFIX}/bench${optimized_suffix}${nsys_suffix}${torch_suffix}_log/vllm_bench_${MODE}${gems_suffix}_logs"
    local server_log_dir="${PATH_PREFIX}/server-logs"
    local nsys_output_dir="${PATH_PREFIX}/nsys-raw"
    local torch_output_dir="${PATH_PREFIX}/torch-raw"
    local reports_dir="${REPORT_PREFIX}"

    yq -i ".paths.log_dir = \"$log_dir\"" "$TOOL_CONFIG"
    yq -i ".paths.server_log_dir = \"$server_log_dir\"" "$TOOL_CONFIG"
    yq -i ".paths.nsys_output_dir = \"$nsys_output_dir\"" "$TOOL_CONFIG"
    yq -i ".paths.torch_output_dir = \"$torch_output_dir\"" "$TOOL_CONFIG"
    yq -i ".paths.reports_dir = \"$reports_dir\"" "$TOOL_CONFIG"
    yq -i ".paths.model_name = \"$MODEL_NAME\"" "$TOOL_CONFIG"

    log_info "工具配置已更新 (路径前缀: ${PATH_PREFIX})"
}

# 应用 vLLM 补丁
apply_vllm_patch() {
    if [[ "$MODE" == "gems" ]]; then
        log_step "应用 vLLM 补丁..."
        "${SCRIPT_DIR}/patch-vllm.sh" --method search
    else
        log_info "CUDA 模式，跳过补丁"
    fi
}

# 还原 vLLM 补丁
restore_vllm_patch() {
    log_step "还原 vLLM 补丁..."
    "${SCRIPT_DIR}/restore-vllm.sh" 2>/dev/null || true
}

# 生成 vllm serve 命令
generate_serve_command() {
    local cmd="vllm serve ${MODEL_PATH}"
    cmd+=" --tensor-parallel-size ${TENSOR_PARALLEL}"
    cmd+=" --port ${PORT}"
    cmd+=" --served-model-name ${MODEL_NAME}"

    # 从配置读取服务参数
    local gpu_mem_util trust_remote reasoning_parser
    gpu_mem_util=$(yq '.serve.gpu_memory_utilization' "$CONFIG_FILE")
    trust_remote=$(yq '.serve.trust_remote_code' "$CONFIG_FILE")
    reasoning_parser=$(yq '.serve.reasoning_parser // ""' "$CONFIG_FILE")
    max_batched_tokens=$(yq '.serve.max_num_batched_tokens // ""' "$CONFIG_FILE")
    max_seqs=$(yq '.serve.max_num_seqs // ""' "$CONFIG_FILE")
    extra_args=$(yq '.serve.extra_args // ""' "$CONFIG_FILE")

    [[ -n "$gpu_mem_util" && "$gpu_mem_util" != "null" ]] && cmd+=" --gpu_memory_utilization $gpu_mem_util"
    [[ "$trust_remote" == "true" ]] && cmd+=" --trust-remote-code"
    [[ -n "$reasoning_parser" && "$reasoning_parser" != "null" ]] && cmd+=" --reasoning-parser $reasoning_parser"
    [[ -n "$max_batched_tokens" && "$max_batched_tokens" != "null" ]] && cmd+=" --max-num-batched-tokens $max_batched_tokens"
    [[ -n "$max_seqs" && "$max_seqs" != "null" ]] && cmd+=" --max-num-seqs $max_seqs"
    [[ -n "$extra_args" && "$extra_args" != "null" ]] && cmd+=" $extra_args"

    echo "$cmd"
}

# 生成环境变量
generate_env_vars() {
    local env_vars=()

    if [[ "$MODE" == "cuda" ]]; then
        # CUDA 模式不设置 FlagGems 环境变量
        :
    elif [[ "$MODE" == "gems" ]]; then
        env_vars+=("USE_FLAGOS=1")
        if [[ "$GEMS_MODE" =~ ^[0-9]+$ ]]; then
            env_vars+=("USE_GEMS_MODE=${GEMS_MODE}")
        else
            env_vars+=("USE_GEMS_MODE=\"${GEMS_MODE}\"")
        fi
        env_vars+=("GEMS_ONCE=true")
    fi

    echo "${env_vars[*]}"
}

# Tmux 会话管理
TMUX_SESSION="screen"
TMUX_WINDOW="bench"

ensure_tmux_session() {
    if ! command -v tmux &>/dev/null; then
        log_error "tmux 未安装"
        exit 1
    fi

    if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${TMUX_SESSION}" -n "${TMUX_WINDOW}"
    else
        if ! tmux list-windows -t "${TMUX_SESSION}" 2>/dev/null | grep -q "${TMUX_WINDOW}"; then
            tmux new-window -t "${TMUX_SESSION}" -n "${TMUX_WINDOW}"
        fi
    fi
}

send_to_tmux() {
    local cmd="$1"
    tmux send-keys -t "${TMUX_SESSION}:${TMUX_WINDOW}" C-c 2>/dev/null || true
    sleep 0.5
    tmux send-keys -t "${TMUX_SESSION}:${TMUX_WINDOW}" "${cmd}" C-m
}

stop_tmux_program() {
    tmux send-keys -t "${TMUX_SESSION}:${TMUX_WINDOW}" C-c 2>/dev/null || true
}

# 等待服务器就绪
wait_server_ready() {
    local timeout=3600
    local interval=2
    local url="http://localhost:${PORT}/v1/models"

    log_step "等待服务器就绪: $url"

    local start_ts now_ts elapsed
    start_ts=$(date +%s)

    while true; do
        if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
            log_info "服务器已就绪"
            return 0
        fi

        now_ts=$(date +%s)
        elapsed=$((now_ts - start_ts))
        if (( elapsed >= timeout )); then
            log_error "等待服务器超时 (${timeout}s)"
            return 1
        fi

        sleep "$interval"
    done
}

# 运行基准测试
run_benchmark() {
    log_step "运行基准测试..."

    # 创建日志目录
    local log_dir
    log_dir=$(yq '.paths.log_dir' "$TOOL_CONFIG")
    mkdir -p "${PROJECT_ROOT}/${log_dir}"

    # 运行 Python 基准测试脚本 (使用当前 Python 环境)
    cd "$PROJECT_ROOT"

    # 使用当前 Python 解释器
    local python_exe="${Python_EXECUTABLE:-python3}"
    $python_exe "${SCRIPT_DIR}/tools/benchmark_runner.py"
}

# 单次运行
run_single() {
    log_step "========== 运行: MODE=$MODE, DEVICE=$DEVICE, GEMS_MODE=$GEMS_MODE =========="

    # 1. 更新配置
    update_tool_config

    # 2. 应用补丁
    apply_vllm_patch

    # 3. 生成命令
    local serve_cmd env_vars
    serve_cmd=$(generate_serve_command)
    env_vars=$(generate_env_vars)

    log_info "Serve 命令: TRITON_PRINT_AUTOTUNING=1 $env_vars $serve_cmd"

    # 4. 在 tmux 中启动服务器
    ensure_tmux_session
    cd "$PROJECT_ROOT"

    # 获取当前 conda 环境激活命令
    local conda_activate=""
    if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        conda_activate="source $(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_DEFAULT_ENV} && "
    fi

    local full_cmd="${conda_activate}cd $PROJECT_ROOT && TRITON_PRINT_AUTOTUNING=1 $env_vars $serve_cmd"
    send_to_tmux "$full_cmd"

    # 5. 等待服务器就绪
    wait_server_ready

    if [[ "$RUN_IDLE" == "true" ]]; then
        log_info "仅启动服务器模式，不运行测试"
        log_info "服务器运行在 tmux 会话: $TMUX_SESSION:$TMUX_WINDOW"
        log_info "连接命令: tmux attach -t $TMUX_SESSION"
        return 0
    fi

    # 6. 运行基准测试
    run_benchmark

    # 7. 停止服务器
    log_step "停止服务器..."
    stop_tmux_program
    sleep 2

    # 8. 还原补丁
    restore_vllm_patch
}

# 批量运行 (逐算子)
run_batch() {
    log_step "批量模式: 逐算子运行"

    # 加载算子列表
    local ops=()
    if [[ -n "$OPS_FILE" && -f "$OPS_FILE" ]]; then
        mapfile -t ops < "$OPS_FILE"
    else
        # 从配置加载
        source "${SCRIPT_DIR}/target_ops.sh"
        ops=("${OPS[@]}")
    fi

    log_info "将运行 ${#ops[@]} 个算子"

    for op in "${ops[@]}"; do
        # 去除逗号和空格
        op=$(echo "$op" | tr -d ', ')

        if [[ -z "$op" ]]; then
            continue
        fi

        log_step "---------- 算子: $op ----------"

        # 设置 GEMS_MODE 为当前算子
        GEMS_MODE="$op"
        run_single

        # 等待一段时间
        log_info "等待 10 秒..."
        sleep 10
    done

    log_info "所有算子测试完成"
}

# 主函数
main() {
    parse_args "$@"

    log_info "FlagTune 工作流调度器"
    log_info "模式: $MODE, 设备: $DEVICE"

    check_dependencies
    read_config

    cd "$PROJECT_ROOT"

    if [[ "$BATCH_MODE" == "true" ]]; then
        run_batch
    else
        run_single
    fi

    log_info "完成!"
}

main "$@"
