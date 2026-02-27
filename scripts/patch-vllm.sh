#!/usr/bin/env bash
#
# patch-vllm.sh - 对 vLLM 的 gpu_model_runner.py 进行动态补丁
#
# 用法:
#   ./patch-vllm.sh                    # 应用补丁
#   ./patch-vllm.sh --restore          # 还原原始文件
#   ./patch-vllm.sh --backup-only      # 仅备份不修改
#   ./patch-vllm.sh --method line --line-num 201  # 使用行号定位
#   ./patch-vllm.sh --method search    # 使用搜索定位 (默认)
#
# 定位方法:
#   search (默认): 搜索特定行，在下一行插入
#   line: 使用指定行号插入
#

set -euo pipefail

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL_CONFIG="${SCRIPT_DIR}/tools/tool_config.yaml"

# 默认参数
RESTORE=false
BACKUP_ONLY=false
METHOD="search"
LINE_NUM=201
SEARCH_PATTERN="PerLayerAttnMetadata: TypeAlias = list\[AttnMetadataDict\] | AttnMetadataDict"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --restore)
            RESTORE=true
            shift
            ;;
        --backup-only)
            BACKUP_ONLY=true
            shift
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --line-num)
            LINE_NUM="$2"
            shift 2
            ;;
        --search-pattern)
            SEARCH_PATTERN="$2"
            shift 2
            ;;
        -h|--help)
            head -20 "$0" | tail -18
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检测 site-packages 路径
detect_site_packages() {
    python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || {
        log_error "无法检测 site-packages 路径"
        exit 1
    }
}

# 查找 gpu_model_runner.py
find_gpu_model_runner() {
    local site_packages="$1"
    local gpu_runner="${site_packages}/vllm/v1/worker/gpu_model_runner.py"

    if [[ -f "$gpu_runner" ]]; then
        echo "$gpu_runner"
        return 0
    fi

    # 尝试其他可能的位置
    for path in "${site_packages}/vllm/v1/worker/gpu_model_runner.py"; do
        if [[ -f "$path" ]]; then
            echo "$path"
            return 0
        fi
    done

    log_error "找不到 gpu_model_runner.py"
    return 1
}

# 还原原始文件
restore_file() {
    local target_file="$1"
    local backup_file="${target_file}.bak"

    if [[ -f "$backup_file" ]]; then
        cp "$backup_file" "$target_file"
        log_info "已从备份还原: $target_file"
        rm -f "$backup_file"
        log_info "已删除备份文件: $backup_file"
    else
        log_warn "备份文件不存在: $backup_file"
    fi
}

# 备份原始文件
backup_file() {
    local target_file="$1"
    local backup_file="${target_file}.bak"

    if [[ -f "$backup_file" ]]; then
        log_warn "备份文件已存在: $backup_file"
    else
        cp "$target_file" "$backup_file"
        log_info "已创建备份: $backup_file"
    fi
}

# 检查是否已打补丁
is_patched() {
    local target_file="$1"
    grep -q "USE_FLAGOS" "$target_file" 2>/dev/null
}

# 生成补丁代码
generate_patch_code() {
    cat << 'PATCH_EOF'
import os
if os.getenv("USE_FLAGOS") == "1":
    import flag_gems

    USE_GEMS_MODE = os.getenv("USE_GEMS_MODE")
    GEMS_ONCE = os.getenv("GEMS_ONCE", "True").lower() == "true"
    GEMS_SAVE_PATH = os.getenv("GEMS_SAVE_PATH")  # 完整的保存目录路径

    # import from FlagGems/src/flag_gems/__init__.py
    FlagGemsList=["_unique2", "_upsample_bicubic2d_aa", "abs", "abs_", "acos",
                  "add", "add_", "addcdiv", "addcmul", "addmm", "addmm_out",
                  "addmv", "addmv_out", "addr", "all", "all_dim", "all_dims",
                  "allclose", "amax", "angle", "any", "any_dim", "any_dims",
                  "arange", "arange_start", "argmax", "argmin", "atan", "atan_",
                  "avg_pool2d", "avg_pool2d_backward", "baddbmm", "batch_norm",
                  "batch_norm_backward", "bitwise_and_scalar", "bitwise_and_scalar_",
                  "bitwise_and_scalar_tensor", "bitwise_and_tensor", "bitwise_and_tensor_",
                  "bitwise_left_shift", "bitwise_not", "bitwise_not_", "bitwise_or_scalar",
                  "bitwise_or_scalar_", "bitwise_or_scalar_tensor", "bitwise_or_tensor",
                  "bitwise_or_tensor_", "bitwise_right_shift", "bmm", "bmm_out", "cat",
                  "celu", "celu_", "clamp", "clamp_", "clamp_min", "clamp_min_",
                  "clamp_tensor", "clamp_tensor_", "constant_pad_nd", "conv1d",
                  "conv2d", "conv3d", "copy_", "cos", "cos_", "count_nonzero",
                  "cummax", "cummin", "cumsum", "cumsum_out", "diag", "diag_embed",
                  "diagonal_backward", "div_mode", "div_mode_", "dot", "dropout",
                  "dropout_backward", "elu", "elu_", "elu_backward", "embedding",
                  "embedding_backward", "eq", "eq_scalar", "erf", "erf_", "exp",
                  "exp2", "exp2_", "exp_", "exp_out", "exponential_", "eye", "eye_m",
                  "fill_scalar", "fill_scalar_", "fill_tensor", "fill_tensor_",
                  "flash_attention_forward", "flip", "floor_divide", "floor_divide_",
                  "full", "full_like", "gather", "gather_backward", "ge", "ge_scalar",
                  "gelu", "gelu_", "gelu_backward", "glu", "glu_backward", "group_norm",
                  "group_norm_backward", "gt", "gt_scalar", "hstack", "index",
                  "index_add", "index_add_", "index_put", "index_put_", "index_select",
                  "isclose", "isfinite", "isin", "isinf", "isnan", "kron", "layer_norm",
                  "layer_norm_backward", "le", "le_scalar", "lerp_scalar", "lerp_scalar_",
                  "lerp_tensor", "lerp_tensor_", "linspace", "log", "log_sigmoid",
                  "log_softmax", "log_softmax_backward", "logical_and", "logical_not",
                  "logical_or", "logical_xor", "logspace", "lt", "lt_scalar", "masked_fill",
                  "masked_fill_", "masked_scatter", "masked_scatter_", "masked_select",
                  "max", "max_dim", "max_pool2d_backward", "max_pool2d_with_indices",
                  "maximum", "mean", "mean_dim", "min", "min_dim", "minimum", "mm",
                  "mm_out", "moe_sum", "mse_loss", "mul", "mul_", "multinomial",
                  "mv", "nan_to_num", "ne", "ne_scalar", "neg", "neg_", "nll_loss2d_backward",
                  "nll_loss2d_forward", "nll_loss_backward", "nll_loss_forward",
                  "nonzero", "normal_float_tensor", "normal_tensor_float", "normal_tensor_tensor",
                  "ones", "ones_like", "pad", "polar", "pow_scalar", "pow_tensor_scalar",
                  "pow_tensor_scalar_", "pow_tensor_tensor", "pow_tensor_tensor_",
                  "prod", "prod_dim", "quantile", "rand", "rand_like", "randn",
                  "randn_like", "randperm", "reciprocal", "reciprocal_", "relu",
                  "relu_", "remainder", "remainder_", "repeat", "repeat_interleave_self_int",
                  "repeat_interleave_self_tensor", "repeat_interleave_tensor",
                  "resolve_conj", "resolve_neg", "rms_norm", "rsqrt", "rsqrt_",
                  "scaled_softmax_backward", "scaled_softmax_forward", "scatter",
                  "scatter_", "scatter_add_", "select_scatter", "sigmoid", "sigmoid_",
                  "sigmoid_backward", "silu", "silu_", "silu_backward", "sin", "sin_",
                  "slice_scatter", "softmax", "softmax_backward", "softplus", "sort",
                  "sort_stable", "sqrt", "sqrt_", "stack", "std", "sub", "sub_", "sum",
                  "sum_dim", "sum_dim_out", "sum_out", "tan", "tan_", "tanh", "tanh_",
                  "tanh_backward", "threshold", "threshold_backward", "tile", "to_copy",
                  "topk", "trace", "triu", "true_divide", "true_divide_", "true_divide_out",
                  "uniform_", "upsample_nearest1d", "upsample_nearest2d", "var_mean",
                  "vdot", "vector_norm", "vstack", "weight_norm_interface",
                  "weight_norm_interface_backward", "where_scalar_other", "where_scalar_self",
                  "where_self", "where_self_out", "zeros", "zeros_like"]

    # 构建保存路径: GEMS_SAVE_PATH/xxx.txt
    def _get_gems_path(filename):
        if GEMS_SAVE_PATH:
            os.makedirs(GEMS_SAVE_PATH, exist_ok=True)
            return f"{GEMS_SAVE_PATH}/{filename}"
        return None

    if USE_GEMS_MODE == "all":
        gems_path = _get_gems_path(f"gems-{USE_GEMS_MODE}.txt")
        kwargs = {"record": True, "once": GEMS_ONCE}
        if gems_path:
            kwargs["path"] = gems_path
        flag_gems.enable(**kwargs)
    elif USE_GEMS_MODE == "NULL":
        gems_path = _get_gems_path(f"gems-{USE_GEMS_MODE}.txt")
        kwargs = {"record": True, "once": GEMS_ONCE, "unused": FlagGemsList}
        if gems_path:
            kwargs["path"] = gems_path
        flag_gems.enable(**kwargs)
    else:
        keep_ops = [USE_GEMS_MODE] if isinstance(USE_GEMS_MODE, str) else USE_GEMS_MODE
        op_name = USE_GEMS_MODE if isinstance(USE_GEMS_MODE, str) else "custom"
        gems_path = _get_gems_path(f"gems-{op_name}-{keep_ops}.txt")
        kwargs = {"record": True, "once": GEMS_ONCE, "include": keep_ops}
        if gems_path:
            kwargs["path"] = gems_path
        flag_gems.only_enable(**kwargs)

PATCH_EOF
}

# 应用补丁 (使用搜索方式)
apply_patch_search() {
    local target_file="$1"
    local patch_code
    patch_code=$(generate_patch_code)

    # 查找匹配行
    local found_line
    found_line=$(grep -n "$SEARCH_PATTERN" "$target_file" | head -1 | cut -d: -f1)

    if [[ -z "$found_line" ]]; then
        log_error "未找到搜索模式: $SEARCH_PATTERN"
        return 1
    fi

    log_info "在行 $found_line 找到目标，将在下一行插入补丁"

    # 创建临时文件
    local temp_file
    temp_file=$(mktemp)

    # 插入补丁
    head -n "$found_line" "$target_file" > "$temp_file"
    echo "" >> "$temp_file"
    echo "$patch_code" >> "$temp_file"
    tail -n +"$((found_line + 1))" "$target_file" >> "$temp_file"

    # 替换原文件
    mv "$temp_file" "$target_file"

    log_info "补丁已应用 (搜索模式)"
}

# 应用补丁 (使用行号方式)
apply_patch_line() {
    local target_file="$1"
    local patch_code
    patch_code=$(generate_patch_code)

    log_info "将在行 $LINE_NUM 插入补丁"

    # 创建临时文件
    local temp_file
    temp_file=$(mktemp)

    # 插入补丁
    head -n "$((LINE_NUM - 1))" "$target_file" > "$temp_file"
    echo "$patch_code" >> "$temp_file"
    tail -n +"$LINE_NUM" "$target_file" >> "$temp_file"

    # 替换原文件
    mv "$temp_file" "$target_file"

    log_info "补丁已应用 (行号模式)"
}

# 应用补丁
apply_patch() {
    local target_file="$1"

    if is_patched "$target_file"; then
        log_warn "文件已打过补丁: $target_file"
        return 0
    fi

    case "$METHOD" in
        search)
            apply_patch_search "$target_file"
            ;;
        line)
            apply_patch_line "$target_file"
            ;;
        *)
            log_error "未知的定位方法: $METHOD"
            return 1
            ;;
    esac
}

# 更新 tool_config.yaml
update_tool_config() {
    local site_packages="$1"
    local gpu_runner="$2"

    if command -v yq &>/dev/null; then
        yq -i ".vllm.site_packages = \"$site_packages\"" "$TOOL_CONFIG"
        yq -i ".vllm.gpu_model_runner = \"$gpu_runner\"" "$TOOL_CONFIG"
    else
        log_warn "yq 未安装，跳过更新 tool_config.yaml"
    fi
}

# 主函数
main() {
    # 检测路径
    local site_packages
    site_packages=$(detect_site_packages)
    log_info "检测到 site-packages: $site_packages"

    local gpu_runner
    gpu_runner=$(find_gpu_model_runner "$site_packages")
    log_info "找到 gpu_model_runner.py: $gpu_runner"

    # 更新配置
    update_tool_config "$site_packages" "$gpu_runner"

    # 还原模式
    if [[ "$RESTORE" == "true" ]]; then
        restore_file "$gpu_runner"
        exit 0
    fi

    # 备份
    backup_file "$gpu_runner"

    # 仅备份模式
    if [[ "$BACKUP_ONLY" == "true" ]]; then
        log_info "仅备份模式，不修改文件"
        exit 0
    fi

    # 应用补丁
    apply_patch "$gpu_runner"

    log_info "完成!"
}

main "$@"
