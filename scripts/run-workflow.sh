#!/usr/bin/env bash
#
# run-workflow.sh - FlagTune 通用工作流调度器
#
# 用法:
#   ./run-workflow.sh --mode cuda --device 0
#   ./run-workflow.sh --mode gems --device 0 --gems-mode all
#   ./run-workflow.sh --mode gems --device 0 --ops-file ops.txt
#   ./run-workflow.sh --mode cuda --device 0 --model deepseek-3.2
#
# 参数:
#   --mode cuda|gems      运行模式
#   --device N            GPU 设备 ID
#   --model NAME          使用 config.yaml.NAME 作为配置文件
#   --gems-mode MODE      FlagGems 模式 (all|NULL|算子名)
#   --ops-file FILE       算子列表文件
#   --optimized           优化模式
#   --nsys                Nsys 性能分析
#   --torch               Torch 性能分析
#   --run-idle            仅启动服务器不运行测试
#   --batch               批量模式 (逐算子运行)
#   --reuse               复用已有服务器，不重新启动
#   --gems-once           GEMS_ONCE 参数 (默认 true)
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
REUSE_SERVER=false  # 复用已有服务器
GEMS_ONCE=true  # GEMS_ONCE 参数
MODEL_CONFIG=""  # 模型配置后缀，如 "deepseek-3.2"

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
            --reuse)
                REUSE_SERVER=true
                shift
                ;;
            --gems-once)
                GEMS_ONCE="$2"
                shift 2
                ;;
            --model)
                MODEL_CONFIG="$2"
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

# 确定配置文件路径
resolve_config_file() {
    if [[ -n "$MODEL_CONFIG" ]]; then
        CONFIG_FILE="${PROJECT_ROOT}/config.yaml.${MODEL_CONFIG}"
        if [[ ! -f "$CONFIG_FILE" ]]; then
            log_error "配置文件不存在: $CONFIG_FILE"
            log_info "请从 config.yaml.template 创建: cp config.yaml.template config.yaml.${MODEL_CONFIG}"
            exit 1
        fi
        log_info "使用模型配置: $CONFIG_FILE"
    else
        CONFIG_FILE="${PROJECT_ROOT}/config.yaml"
        if [[ ! -f "$CONFIG_FILE" ]]; then
            log_error "默认配置文件不存在: $CONFIG_FILE"
            log_info "请从 config.yaml.template 创建: cp config.yaml.template config.yaml"
            log_info "或使用 --model 参数指定模型配置: --model <name>"
            exit 1
        fi
    fi
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

    # 更新运行配置
    yq -i ".current_run.mode = \"$MODE\"" "$TOOL_CONFIG"
    yq -i ".current_run.device = $DEVICE" "$TOOL_CONFIG"
    yq -i ".current_run.port = $PORT" "$TOOL_CONFIG"
    yq -i ".current_run.gems.mode = \"$GEMS_MODE\"" "$TOOL_CONFIG"
    yq -i ".current_run.optimized = $OPTIMIZED" "$TOOL_CONFIG"
    yq -i ".current_run.nsys_profile = $NSYS_PROFILE" "$TOOL_CONFIG"
    yq -i ".current_run.torch_profile = $TORCH_PROFILE" "$TOOL_CONFIG"

    # 从 CONFIG_FILE 复制模型配置到 tool_config
    local model_path model_name tokenizer_path tensor_parallel
    model_path=$(yq '.model.path' "$CONFIG_FILE")
    model_name=$(yq '.model.name' "$CONFIG_FILE")
    tokenizer_path=$(yq '.model.tokenizer_path // ""' "$CONFIG_FILE")
    tensor_parallel=$(yq '.serve.tensor_parallel_size // .model.tensor_parallel_size // 8' "$CONFIG_FILE")

    yq -i ".model.path = \"$model_path\"" "$TOOL_CONFIG"
    yq -i ".model.name = \"$model_name\"" "$TOOL_CONFIG"
    yq -i ".model.tokenizer_path = \"$tokenizer_path\"" "$TOOL_CONFIG"
    yq -i ".model.tensor_parallel_size = $tensor_parallel" "$TOOL_CONFIG"

    # 从 CONFIG_FILE 复制基准测试配置到 tool_config
    local benchmark_host benchmark_num_runs
    benchmark_host=$(yq '.benchmark.host // "127.0.0.1"' "$CONFIG_FILE")
    benchmark_num_runs=$(yq '.benchmark.num_runs // 4' "$CONFIG_FILE")

    yq -i ".benchmark.host = \"$benchmark_host\"" "$TOOL_CONFIG"
    yq -i ".benchmark.num_runs = $benchmark_num_runs" "$TOOL_CONFIG"

    # 复制场景配置
    yq -i ".benchmark.scenarios = $(yq '.benchmark.scenarios' "$CONFIG_FILE" -o=json)" "$TOOL_CONFIG"

    # 构建 shape 后缀 (GEMS_ONCE=false 时添加)
    local shape_suffix=""
    if [[ "$GEMS_ONCE" == "false" ]]; then
        shape_suffix="-shape"
    fi

    # 更新日志路径 (包含模型名和 shape 后缀)
    local log_dir="${PATH_PREFIX}/bench${optimized_suffix}${nsys_suffix}${torch_suffix}_log${shape_suffix}/vllm_bench_${MODE}${gems_suffix}_logs"
    local server_log_dir="${PATH_PREFIX}/server-logs${shape_suffix}"
    local nsys_output_dir="${PATH_PREFIX}/nsys-raw${shape_suffix}"
    local torch_output_dir="${PATH_PREFIX}/torch-raw${shape_suffix}"
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

# 生成服务端日志文件路径
generate_server_log_file() {
    local server_log_dir
    server_log_dir=$(yq '.paths.server_log_dir' "$TOOL_CONFIG")

    # 构建后缀
    local gems_suffix=""
    [[ "$MODE" == "gems" ]] && gems_suffix="_${GEMS_MODE}"

    local optimized_suffix=""
    [[ "$OPTIMIZED" == "true" ]] && optimized_suffix="_optimized"

    local nsys_suffix=""
    [[ "$NSYS_PROFILE" == "true" ]] && nsys_suffix="_nsys_profile"

    local torch_suffix=""
    [[ "$TORCH_PROFILE" == "true" ]] && torch_suffix="_torch_profile"

    # 构建 shape 后缀 (GEMS_ONCE=false 时添加)
    local shape_suffix=""
    if [[ "$GEMS_ONCE" == "false" ]]; then
        shape_suffix="-shape"
    fi

    # 创建目录
    mkdir -p "${PROJECT_ROOT}/${server_log_dir}"

    # 返回日志文件路径
    echo "${PROJECT_ROOT}/${server_log_dir}/vllm_bench_${MODE}${gems_suffix}${optimized_suffix}${nsys_suffix}${torch_suffix}${shape_suffix}_server.log"
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
    tokenizer_mode=$(yq '.serve.tokenizer_mode // ""' "$CONFIG_FILE")
    reasoning_parser=$(yq '.serve.reasoning_parser // ""' "$CONFIG_FILE")
    tool_call_parser=$(yq '.serve.tool_call_parser // ""' "$CONFIG_FILE")
    max_batched_tokens=$(yq '.serve.max_num_batched_tokens // ""' "$CONFIG_FILE")
    max_seqs=$(yq '.serve.max_num_seqs // ""' "$CONFIG_FILE")
    extra_args=$(yq '.serve.extra_args // ""' "$CONFIG_FILE")

    [[ -n "$gpu_mem_util" && "$gpu_mem_util" != "null" ]] && cmd+=" --gpu_memory_utilization $gpu_mem_util"
    [[ "$trust_remote" == "true" ]] && cmd+=" --trust-remote-code"
    [[ -n "$tokenizer_mode" && "$tokenizer_mode" != "null" ]] && cmd+=" --tokenizer-mode $tokenizer_mode"
    [[ -n "$reasoning_parser" && "$reasoning_parser" != "null" ]] && cmd+=" --reasoning-parser $reasoning_parser"
    [[ -n "$tool_call_parser" && "$tool_call_parser" != "null" ]] && cmd+=" --tool-call-parser $tool_call_parser"
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
        env_vars+=("GEMS_ONCE=${GEMS_ONCE}")

        # 构建保存路径: 根据 GEMS_ONCE 选择目录名
        local config_dir="gems-config"
        if [[ "$GEMS_ONCE" == "false" ]]; then
            config_dir="gems-config-shape"
        fi
        env_vars+=("GEMS_SAVE_PATH=\"./results/${MODEL_NAME}/${config_dir}\"")

        # 读取 gems.unuse 配置并设置 GEMS_UNUSE 环境变量
        if command -v yq &>/dev/null && [[ -f "$CONFIG_FILE" ]]; then
            local gems_unuse
            gems_unuse=$(yq '.gems.unuse // []' "$CONFIG_FILE" -o=json 2>/dev/null)
            if [[ -n "$gems_unuse" && "$gems_unuse" != "null" && "$gems_unuse" != "[]" ]]; then
                env_vars+=("GEMS_UNUSE='${gems_unuse}'")
            fi
        fi
    fi

    echo "${env_vars[*]}"
}

# Tmux 会话管理
TMUX_SESSION="screen"
TMUX_WINDOW="bench"

# 获取环境激活命令 (支持 conda 和 uv/venv)
get_env_activate_cmd() {
    local activate_cmd=""

    # 优先检测 conda 环境
    if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        local conda_base
        conda_base=$(conda info --base 2>/dev/null)
        if [[ -n "$conda_base" ]]; then
            activate_cmd="source ${conda_base}/etc/profile.d/conda.sh && conda activate ${CONDA_DEFAULT_ENV} && "
            log_info "检测到 conda 环境: ${CONDA_DEFAULT_ENV}"
        fi
    # 检测 uv/venv 虚拟环境
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        local activate_script="${VIRTUAL_ENV}/bin/activate"
        if [[ -f "$activate_script" ]]; then
            activate_cmd="source ${activate_script} && "
            log_info "检测到虚拟环境: ${VIRTUAL_ENV}"
        fi
    fi

    echo "$activate_cmd"
}

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
    local benchmark_cmd="$python_exe ${SCRIPT_DIR}/tools/benchmark_runner.py"

    log_info "执行命令: $benchmark_cmd"
    eval "$benchmark_cmd"
}

# 生成 profiling 命令
generate_profile_command() {
    local python_exe="${Python_EXECUTABLE:-python3}"
    local benchmark_cmd="$python_exe ${SCRIPT_DIR}/tools/benchmark_runner.py"

    # 获取环境变量
    local env_vars
    env_vars=$(generate_env_vars)

    # 构建基础环境变量前缀
    local env_prefix="TRITON_PRINT_AUTOTUNING=1"
    if [[ -n "$env_vars" ]]; then
        env_prefix="$env_prefix $env_vars"
    fi

    # 构建 nsys 输出路径
    if [[ "$NSYS_PROFILE" == "true" ]]; then
        local nsys_output_dir
        nsys_output_dir=$(yq '.paths.nsys_output_dir' "$TOOL_CONFIG")
        mkdir -p "${PROJECT_ROOT}/${nsys_output_dir}"

        local gems_suffix=""
        [[ "$MODE" == "gems" ]] && gems_suffix="_${GEMS_MODE}"
        local nsys_output="${nsys_output_dir}/report_${MODE}${gems_suffix}"

        # log_info "启用 nsys profiling，输出: ${nsys_output}"

        # 命令顺序: env_vars → nsys → python
        echo "${env_prefix} nsys profile -t cuda,nvtx,osrt -o \"${nsys_output}\" --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --force-overwrite true ${benchmark_cmd}"
    elif [[ "$TORCH_PROFILE" == "true" ]]; then
        local torch_output_dir
        torch_output_dir=$(yq '.paths.torch_output_dir' "$TOOL_CONFIG")
        mkdir -p "${PROJECT_ROOT}/${torch_output_dir}"
        # log_info "启用 torch profiling，输出目录: ${torch_output_dir}"

        # torch profiling 在 benchmark_runner.py 内部处理
        echo "${env_prefix} ${benchmark_cmd}"
    else
        echo "${benchmark_cmd}"
    fi
}

# 等待 tmux 命令完成
wait_tmux_command_done() {
    local timeout=7200  # 2 hours timeout
    local interval=5
    local start_ts now_ts elapsed

    start_ts=$(date +%s)
    log_step "等待 profiling 命令完成..."

    while true; do
        # 检查 tmux 窗口中是否还有进程在运行 (除了 shell 本身)
        local pane_pid
        pane_pid=$(tmux list-panes -t "${TMUX_SESSION}:${TMUX_WINDOW}" -F "#{pane_pid}" 2>/dev/null | head -1)

        if [[ -n "$pane_pid" ]]; then
            # 检查是否有子进程 (除了 bash/sh/zsh)
            local child_count
            child_count=$(pgrep -P "$pane_pid" 2>/dev/null | wc -l | tr -d '[:space:]')

            # 如果没有子进程，说明命令已完成
            if [[ "$child_count" -eq 0 ]]; then
                log_info "Profiling 命令已完成"
                return 0
            fi
        fi

        now_ts=$(date +%s)
        elapsed=$((now_ts - start_ts))
        if (( elapsed >= timeout )); then
            log_error "等待 profiling 完成超时 (${timeout}s)"
            return 1
        fi

        sleep "$interval"
    done
}

# 单次运行
run_single() {
    log_step "========== 运行: MODE=$MODE, DEVICE=$DEVICE, GEMS_MODE=$GEMS_MODE =========="

    # 1. 更新配置
    update_tool_config

    # 检查是否启用 profiling 模式
    if [[ "$NSYS_PROFILE" == "true" || "$TORCH_PROFILE" == "true" ]]; then
        # Profiling 模式: 在 tmux 中运行 vllm bench throughput
        log_info "Profiling 模式: 使用 vllm bench throughput (内置服务器)"

        # 应用补丁 (gems 模式需要)
        apply_vllm_patch

        # 生成 profiling 命令
        local profile_cmd
        profile_cmd=$(generate_profile_command)
        log_info "Profiling 命令: $profile_cmd"

        # 在 tmux 中启动 profiling
        ensure_tmux_session
        cd "$PROJECT_ROOT"

        # 获取环境激活命令 (支持 conda 和 uv/venv)
        local env_activate
        env_activate=$(get_env_activate_cmd)

        local full_cmd="${env_activate}cd $PROJECT_ROOT && $profile_cmd"
        send_to_tmux "$full_cmd"

        # 等待 profiling 完成
        wait_tmux_command_done

        # 还原补丁
        restore_vllm_patch
    elif [[ "$REUSE_SERVER" == "true" ]]; then
        log_info "复用模式: 跳过服务器启动和补丁操作"

        # 检查服务器是否可用
        local url="http://localhost:${PORT}/v1/models"
        if ! curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
            log_error "服务器不可用: $url"
            log_error "请先启动服务器或移除 --reuse 选项"
            exit 1
        fi
        log_info "服务器已就绪: $url"

        # 运行基准测试
        run_benchmark

        log_info "复用模式: 跳过服务器停止"
    else
        # 正常模式: 启动服务器 -> 运行测试 -> 停止服务器

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

        # 获取环境激活命令 (支持 conda 和 uv/venv)
        local env_activate
        env_activate=$(get_env_activate_cmd)

        # 生成服务端日志文件
        local server_log_file
        server_log_file=$(generate_server_log_file)
        log_info "服务端日志: $server_log_file"

        local full_cmd="${env_activate}cd $PROJECT_ROOT && TRITON_PRINT_AUTOTUNING=1 $env_vars $serve_cmd 2>&1 | tee \"$server_log_file\""
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
    fi
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
    resolve_config_file
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
