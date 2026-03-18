#!/usr/bin/env bash
#
# run-processing.sh - FlagTune 数据处理工作流调度器
#
# 用法:
#   ./run-processing.sh --workflow bench -f optimized
#   ./run-processing.sh --workflow nsys --model qwen-3.5
#   ./run-processing.sh --workflow nsys --model qwen-3.5 --skip-export
#   ./run-processing.sh --workflow torch --model qwen-3.5
#   ./run-processing.sh --workflow torch --model qwen-3.5 --mode cuda
#   ./run-processing.sh --workflow shape
#   ./run-processing.sh --workflow shape --gems-mode mm
#   ./run-processing.sh --workflow shape --report
#   ./run-processing.sh --workflow all --model qwen-3.5
#
# 参数:
#   --workflow bench|nsys|torch|shape|all  工作流选择
#   --model NAME                     使用 config.yaml.NAME 作为配置文件
#   --mode TYPE                      Torch 工作流模式 (cuda|gems|compare，默认 cuda)
#   --gems-mode MODE                 Shape 工作流使用的 FlagGems 模式 (默认 all)
#   -f FILENAME                      基准测试模式 (optimized 或空)
#   --skip-export                    跳过 nsys 导出步骤 (仅 nsys 工作流)
#   --report                         仅执行报告生成相关逻辑
#   --warmup N                       跳过的预热轮数 (默认 2)
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

# Check for yq
if ! command -v yq &>/dev/null; then
    log_error "yq 未安装，请先运行: ./scripts/setup-deps.sh"
    exit 1
fi

# 默认参数
WORKFLOW=""
MODEL_CONFIG=""
FILENAME=""
SKIP_EXPORT=false
REPORT=false
WARMUP=2
GEMS_MODE="all"
TORCH_MODE="cuda"

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --workflow)
                WORKFLOW="$2"
                shift 2
                ;;
            --model)
                MODEL_CONFIG="$2"
                shift 2
                ;;
            --mode)
                TORCH_MODE="$2"
                shift 2
                ;;
            --gems-mode)
                GEMS_MODE="$2"
                shift 2
                ;;
            -f)
                FILENAME="$2"
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
                head -27 "$0" | tail -25
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
            log_warn "默认配置文件不存在: $CONFIG_FILE"
            log_info "将使用 tool_config.yaml 中的路径配置"
        fi
    fi
}

# 检查依赖
check_dependencies() {
    local missing=()
    local required=(python3)

    for cmd in "${required[@]}"; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "缺少必需依赖: ${missing[*]}"
        exit 1
    fi
}

# 读取配置到 tool_config.yaml
update_tool_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_info "跳过配置更新，使用现有 tool_config.yaml"
        return 0
    fi

    log_step "更新工具配置..."

    # 从配置文件读取路径
    local model_name path_prefix report_prefix
    model_name=$(yq '.model.name // "default"' "$CONFIG_FILE")
    local use_model_name=$(yq '.paths.use_model_name // true' "$CONFIG_FILE")
    local paths_results=$(yq '.paths.results // "results"' "$CONFIG_FILE")
    local paths_reports=$(yq '.paths.reports // "reports"' "$CONFIG_FILE")

    if [[ "$use_model_name" == "true" ]]; then
        path_prefix="${paths_results}/${model_name}"
        report_prefix="${paths_reports}/${model_name}"
    else
        path_prefix="${paths_results}"
        report_prefix="${paths_reports}"
    fi

    # 更新 tool_config.yaml 中的路径
    yq -i ".paths.model_name = \"$model_name\"" "$TOOL_CONFIG"
    yq -i ".paths.results = \"$paths_results\"" "$TOOL_CONFIG"
    yq -i ".paths.log_dir = \"${path_prefix}/bench_log\"" "$TOOL_CONFIG"
    yq -i ".paths.reports_dir = \"$report_prefix\"" "$TOOL_CONFIG"
    yq -i ".paths.nsys_output_dir = \"${path_prefix}/nsys-raw\"" "$TOOL_CONFIG"
    yq -i ".paths.torch_output_dir = \"${path_prefix}/torch-raw\"" "$TOOL_CONFIG"
    yq -i ".paths.gems_config_dir = \"${path_prefix}/gems-config\"" "$TOOL_CONFIG"
    yq -i ".paths.gems_config_shape_dir = \"${path_prefix}/gems-config-shape\"" "$TOOL_CONFIG"

    log_info "工具配置已更新 (模型: ${model_name})"
}

# 获取配置值
get_config_value() {
    local key="$1"
    local default="${2:-}"

    if [[ -f "$TOOL_CONFIG" ]]; then
        local value=$(yq ".$key // \"$default\"" "$TOOL_CONFIG")
        echo "$value"
    else
        echo "$default"
    fi
}

# 运行基准测试统计
run_bench_workflow() {
    log_step "运行基准测试统计工作流..."

    local python_exe="${Python_EXECUTABLE:-python3}"
    local bench_script="${SCRIPT_DIR}/tools/bench_stat.py"

    if [[ ! -f "$bench_script" ]]; then
        log_error "脚本不存在: $bench_script"
        exit 1
    fi

    # 从配置获取路径 (参照 run-workflow.sh 的目录设定)
    local model_name=$(get_config_value "paths.model_name" "default")
    local paths_results=$(get_config_value "paths.results" "results")
    local reports_dir=$(get_config_value "paths.reports_dir" "reports/${model_name}")

    # 根据 -f 参数确定日志目录
    # Pattern: results/{model}/bench_{filename}_log or results/{model}/bench_log
    local log_dir
    if [[ -n "$FILENAME" ]]; then
        log_dir="${paths_results}/${model_name}/bench_${FILENAME}_log"
    else
        log_dir=$(get_config_value "paths.log_dir" "${paths_results}/${model_name}/bench_log")
    fi

    log_info "日志目录: $log_dir"
    log_info "报告目录: $reports_dir"

    # 构建参数
    local args=()
    if [[ -n "$FILENAME" ]]; then
        args+=("-f" "$FILENAME")
    fi
    args+=("--warmup" "$WARMUP")

    log_info "运行: $python_exe $bench_script ${args[*]}"
    cd "$PROJECT_ROOT"
    $python_exe "$bench_script" "${args[@]}"

    log_info "基准测试统计完成"
}

# 运行 Nsys 分析工作流
run_nsys_workflow() {
    log_step "运行 Nsys 分析工作流..."

    # 从配置获取路径 (参照 run-workflow.sh 的目录设定)
    local model_name=$(get_config_value "paths.model_name" "default")
    local paths_results=$(get_config_value "paths.results" "results")
    local nsys_output_dir="${paths_results}/${model_name}/nsys-raw"

    # 1. 导出 nsys 结果
    if [[ "$SKIP_EXPORT" == true ]]; then
        log_info "步骤 1/2: 跳过 Nsys 导出 (--skip-export)"
    else
        log_info "步骤 1/2: 导出 Nsys 结果..."
        local export_script="${SCRIPT_DIR}/export-nsys.sh"
        if [[ -f "$export_script" ]]; then
            # 传递 nsys 目录参数
            bash "$export_script" --dir "$nsys_output_dir"
        else
            log_warn "export-nsys.sh 不存在，跳过导出步骤"
        fi
    fi

    # 2. 运行性能分析
    log_info "步骤 2/2: 运行性能分析..."
    local python_exe="${Python_EXECUTABLE:-python3}"
    local perf_script="${SCRIPT_DIR}/tools/perf_analysis.py"

    if [[ ! -f "$perf_script" ]]; then
        log_error "脚本不存在: $perf_script"
        exit 1
    fi

    log_info "运行: $python_exe $perf_script --nsys_path $nsys_output_dir"
    cd "$PROJECT_ROOT"
    $python_exe "$perf_script" --nsys_path "$nsys_output_dir"

    log_info "Nsys 分析完成"
}

# 运行 Torch profiler 分析工作流
run_torch_workflow() {
    log_step "运行 Torch profiler 分析工作流..."

    local model_name=$(get_config_value "paths.model_name" "default")
    local paths_results=$(get_config_value "paths.results" "results")
    local torch_output_dir
    torch_output_dir=$(get_config_value "paths.torch_output_dir" "${paths_results}/${model_name}/torch-raw")

    local python_exe="${Python_EXECUTABLE:-python3}"
    local perf_script="${SCRIPT_DIR}/tools/perf_analysis_torch.py"

    if [[ ! -f "$perf_script" ]]; then
        log_error "脚本不存在: $perf_script"
        exit 1
    fi

    case "$TORCH_MODE" in
        cuda|gems|compare)
            ;;
        *)
            log_error "--mode 仅支持: cuda, gems, compare；当前值: $TORCH_MODE"
            exit 1
            ;;
    esac

    log_info "Torch 模式: $TORCH_MODE"
    log_info "运行: $python_exe $perf_script --torch_path $torch_output_dir --mode $TORCH_MODE"
    cd "$PROJECT_ROOT"
    $python_exe "$perf_script" --torch_path "$torch_output_dir" --mode "$TORCH_MODE"

    log_info "Torch profiler 分析完成"
}

# 运行 Shape 分析工作流
run_shape_workflow() {
    log_step "运行 Shape 分析工作流..."

    local python_exe="${Python_EXECUTABLE:-python3}"
    local shape_script="${SCRIPT_DIR}/tools/gems_shape_info.py"

    if [[ ! -f "$shape_script" ]]; then
        log_error "脚本不存在: $shape_script"
        exit 1
    fi

    # 从配置获取输入输出路径 (参照 run-workflow.sh 的目录设定)
    local model_name=$(get_config_value "paths.model_name" "default")
    local paths_results=$(get_config_value "paths.results" "results")
    local input_name="gems-${GEMS_MODE}.txt"

    # Shape 分析使用 gems-config-shape 目录 (GEMS_ONCE=false 时生成)
    local input_file="${paths_results}/${model_name}/gems-config-shape/${input_name}"
    local marker_file="${paths_results}/${model_name}/gems-config-shape/marker.txt"
    local output_dir="reports/${model_name}/shape"

    # 检查输入文件是否存在
    if [[ ! -f "${PROJECT_ROOT}/${input_file}" ]]; then
        log_warn "输入文件不存在: ${input_file}"
        if [[ "$GEMS_MODE" != "all" ]]; then
            local fallback_input="${paths_results}/${model_name}/gems-config-shape/gems-all.txt"
            if [[ -f "${PROJECT_ROOT}/${fallback_input}" ]]; then
                log_info "尝试回退到: ${fallback_input}"
                input_file="$fallback_input"
            else
                log_info "尝试使用默认路径: gems-config/gems-all.txt"
                input_file="gems-config/gems-all.txt"
            fi
        else
            log_info "尝试使用默认路径: gems-config/gems-all.txt"
            input_file="gems-config/gems-all.txt"
        fi
    fi

    if [[ ! -f "${PROJECT_ROOT}/${marker_file}" ]]; then
        log_warn "marker 文件不存在: ${marker_file}"
        log_info "降级为单文件统计输出: reports/${model_name}/gems_shape_info.txt"
        local output_file="reports/${model_name}/gems_shape_info.txt"
        log_info "运行: $python_exe $shape_script --input $input_file --output $output_file"
        cd "$PROJECT_ROOT"
        $python_exe "$shape_script" --input "$input_file" --output "$output_file"
        log_info "Shape 分析完成"
        return
    fi

    log_info "运行: $python_exe $shape_script --input $input_file --marker $marker_file --output-dir $output_dir"
    cd "$PROJECT_ROOT"
    $python_exe "$shape_script" --input "$input_file" --marker "$marker_file" --output-dir "$output_dir"

    log_info "Shape 分析完成"
}

# 生成指定场景的 bench 统计报告
run_bench_report() {
    local scenario="$1"

    log_step "生成 ${scenario} 场景 bench 报告..."

    local old_filename="$FILENAME"
    FILENAME="$scenario"
    run_bench_workflow
    FILENAME="$old_filename"

    log_info "${scenario} 场景报告生成完成"
}

# 运行所有工作流
run_all_workflows() {
    log_step "运行所有工作流..."

    run_bench_workflow
    echo ""
    run_nsys_workflow
    echo ""
    run_shape_workflow

    log_info "所有工作流完成"
}

run_report_only_workflow() {
    log_step "仅执行报告生成流程..."

    case "$WORKFLOW" in
        bench)
            run_bench_workflow
            ;;
        shape)
            run_bench_report "shape"
            ;;
        torch)
            run_torch_workflow
            ;;
        all)
            run_bench_workflow
            echo ""
            run_bench_report "shape"
            ;;
        nsys)
            log_warn "--report 当前未定义 nsys 专用报告流程，未执行任何操作"
            ;;
    esac
}

# 验证工作流参数
validate_workflow() {
    if [[ -z "$WORKFLOW" ]]; then
        log_error "必须指定 --workflow 参数"
        log_info "可用的工作流: bench, nsys, torch, shape, all"
        exit 1
    fi

    case "$WORKFLOW" in
        bench|nsys|torch|shape|all)
            ;;
        *)
            log_error "未知的工作流: $WORKFLOW"
            log_info "可用的工作流: bench, nsys, torch, shape, all"
            exit 1
            ;;
    esac
}

# 主函数
main() {
    parse_args "$@"

    log_info "FlagTune 数据处理工作流调度器"
    log_info "工作流: ${WORKFLOW:-未指定}"
    log_info "Gems Mode: ${GEMS_MODE}"
    if [[ "$REPORT" == true ]]; then
        log_info "模式: 仅生成报告 (--report)"
    fi

    validate_workflow
    check_dependencies
    resolve_config_file
    update_tool_config

    cd "$PROJECT_ROOT"

    if [[ "$REPORT" == true ]]; then
        run_report_only_workflow
        log_info "完成!"
        return 0
    fi

    case "$WORKFLOW" in
        bench)
            run_bench_workflow
            ;;
        nsys)
            run_nsys_workflow
            ;;
        torch)
            run_torch_workflow
            ;;
        shape)
            run_shape_workflow
            ;;
        all)
            run_all_workflows
            ;;
    esac

    log_info "完成!"
}

main "$@"
