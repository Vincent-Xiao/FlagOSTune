# FlagTune 用户文档

> vLLM + FlagGems 算子评估自动化框架

---

## 简介

FlagTune 是一套用于自动化评估 vLLM + FlagGems 算子性能的脚本框架。它支持多种工作流模式，帮助您快速完成性能基准测试和分析。此外，脚本搭配了 Skills 支持，允许配合 Claude Code 等 Agent 工具辅助调用脚本并获得调试等帮助。

### 核心功能

- **自动化工作流**: 一键完成服务启动、基准测试、结果收集
- **多种测试模式**: Optimized (快速)、Nsys Profile (深度分析)、Shape Analysis (算子形状分析)
- **灵活配置**: 通过 YAML 配置文件管理模型、服务参数和测试场景
- **结果管理**: 按模型名称自动组织结果和报告
- **算子级测试**: 支持逐个算子进行性能对比
- **批量处理**: 支持一键批量执行多种测试模式

---

## 快速开始

### 1. 安装依赖

```bash
./scripts/setup-deps.sh
```

这将安装以下依赖:
- `jq` - JSON 处理工具
- `tmux` - 会话管理
- `yq` - YAML 处理工具
- `pyyaml` - Python YAML 库
- `nsys` - NVIDIA Nsight Systems (可选)

检查依赖状态:
```bash
./scripts/setup-deps.sh --check
```

### 2. 配置模型

编辑 `config.yaml`，修改以下关键配置:

```yaml
model:
  path: /models/YOUR-MODEL-PATH    # 模型路径
  name: your-model-name             # 模型名称 (用于结果目录)

serve:
  tensor_parallel_size: 8           # GPU 数量
  gpu_memory_utilization: 0.9       # GPU 显存利用率
```

### 3. 运行测试

#### 一键自动工作流 (推荐)

```bash
# 运行优化模式 (CUDA + GEMS 对比)
./scripts/auto-workflow.sh --model your-model --device 0 --optimized

# 运行所有模式 (shape -> optimized -> nsys)
./scripts/auto-workflow.sh --model your-model --device 0 --all

# 仅运行 nsys 分析
./scripts/auto-workflow.sh --model your-model --device 0 --nsys
```

#### 手动分步执行

```bash
# 快速测试 (CUDA 基准)
./scripts/run-workflow.sh --mode cuda --device 0 --optimized

# 快速测试 (Gems 基准)
./scripts/run-workflow.sh --mode gems --device 0 --optimized

# 逐算子批量测试
./scripts/run-workflow.sh --mode gems --device 0 --optimized --batch
```

---

## 配置说明

### config.yaml 结构

```yaml
# 模型配置
model:
  path: /models/Qwen3.5-397B-A17B
  name: qwem3.5
  tensor_parallel_size: 8
  tokenizer_path: null              # null 表示使用 model.path

# 服务配置
serve:
  gpu_memory_utilization: 0.9
  max_num_batched_tokens: 16384
  max_num_seqs: 2048
  trust_remote_code: true
  reasoning_parser: qwen3
  extra_args: ""

# 基准测试配置
benchmark:
  host: 127.0.0.1
  port_base: 2345                   # 实际端口 = port_base + device
  num_runs: 2

  scenarios:
    optimized:                      # 快速测试场景
      - name: p4096d1024
        input_len: 4096
        output_len: 1024
        concurrency: 100

    full:                           # 完整测试场景
      - name: p128d128
        input_len: 128
        output_len: 128
        concurrency: 100
      # ... 更多场景

# 目标算子列表
target_ops:
  - layer_norm
  - softmax
  - gather
  # ...

# 路径配置
paths:
  results: results
  reports: reports
  use_model_name: true              # 结果按模型名分组
```

---

## 工作流详解

### 1. 一键自动工作流 (auto-workflow.sh)

**用途**: 高级一键测试脚本，自动编排多个测试步骤

```bash
# 优化模式 (CUDA + GEMS 对比测试)
./scripts/auto-workflow.sh --model your-model --device 0 --optimized

# 形状分析模式 (收集算子形状信息)
./scripts/auto-workflow.sh --model your-model --device 0 --shape

# NSYS 性能分析
./scripts/auto-workflow.sh --model your-model --device 0 --nsys

# 仅 CUDA 的 NSYS 分析
./scripts/auto-workflow.sh --model your-model --device 0 --nsys --cuda

# 仅 GEMS 的 NSYS 分析
./scripts/auto-workflow.sh --model your-model --device 0 --nsys --gems

# 运行所有模式 (shape -> optimized -> nsys)
./scripts/auto-workflow.sh --model your-model --device 0 --all
```

**参数说明**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model NAME` | 模型配置名称 (必需) | - |
| `--device N` | GPU 设备 ID | 0 |
| `--optimized` | 运行优化模式 (CUDA + GEMS) | false |
| `--nsys` | 运行 NSYS 分析 | false |
| `--cuda` | NSYS 模式下仅运行 CUDA | false |
| `--gems` | NSYS 模式下仅运行 GEMS | false |
| `--shape` | 运行形状分析模式 | false |
| `--all` | 运行所有模式 | false |

---

### 2. 核心工作流 (run-workflow.sh)

**用途**: 通用的基准测试工作流调度器

```bash
# CUDA 快速测试
./scripts/run-workflow.sh --mode cuda --device 0 --optimized

# GEMS 所有算子测试
./scripts/run-workflow.sh --mode gems --device 0 --optimized

# GEMS 单个算子测试
./scripts/run-workflow.sh --mode gems --device 0 --optimized --gems-mode layer_norm

# GEMS 逐算子批量测试
./scripts/run-workflow.sh --mode gems --device 0 --optimized --batch

# NSYS 性能分析
./scripts/run-workflow.sh --mode cuda --device 0 --nsys

# Torch Profiler 分析
./scripts/run-workflow.sh --mode cuda --device 0 --torch

# 仅启动服务 (不运行测试)
./scripts/run-workflow.sh --mode cuda --device 0 --run-idle

# 复用已有服务
./scripts/run-workflow.sh --mode gems --device 0 --optimized --reuse
```

**参数说明**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode cuda\|gems` | 运行模式 | cuda |
| `--device N` | GPU 设备 ID | 0 |
| `--model NAME` | 使用 config.yaml.NAME 作为配置文件 | - |
| `--gems-mode MODE` | FlagGems 模式: all, NULL, 或算子名 | all |
| `--ops-file FILE` | 算子列表文件 (用于批量模式) | - |
| `--optimized` | 使用优化场景 (p4096d1024) | false |
| `--nsys` | 启用 NSYS 性能分析 | false |
| `--torch` | 启用 Torch Profiler | false |
| `--run-idle` | 仅启动服务，不运行测试 | false |
| `--batch` | 批量逐算子运行 | false |
| `--reuse` | 复用已存在的服务 | false |
| `--gems-once true\|false` | GEMS_ONCE 参数 | true |
| `--custom-suffix SUFFIX` | 自定义日志路径后缀 | - |

---

### 3. 数据处理工作流

#### 一键自动处理 (auto-processing.sh)

```bash
# 处理优化模式结果
./scripts/auto-processing.sh --model your-model --optimized

# 处理形状分析结果
./scripts/auto-processing.sh --model your-model --shape

# 处理 NSYS 结果
./scripts/auto-processing.sh --model your-model --nsys

# 处理所有结果
./scripts/auto-processing.sh --model your-model --all
```

#### 手动数据处理 (run-processing.sh)

```bash
# 处理基准测试结果
./scripts/run-processing.sh --workflow bench --model your-model -f optimized

# 处理 NSYS 结果 (自动导出并分析)
./scripts/run-processing.sh --workflow nsys --model your-model

# 处理形状分析
./scripts/run-processing.sh --workflow shape --model your-model

# 跳过 NSYS 导出步骤
./scripts/run-processing.sh --workflow nsys --model your-model --skip-export

# 跳过指定数量的 warmup 轮次
./scripts/run-processing.sh --workflow bench --model your-model -f optimized --warmup 1
```

---

### 4. vLLM 补丁管理

#### 为什么需要补丁?

在 Gems 模式下，需要修改 vLLM 的 `gpu_model_runner.py` 来启用 FlagGems 算子。

#### 补丁操作

```bash
# 查看补丁状态
./scripts/restore-vllm.sh --status

# 应用补丁 (推荐方式)
./scripts/patch-vllm.sh --method search

# 仅创建备份
./scripts/patch-vllm.sh --backup-only

# 还原补丁
./scripts/restore-vllm.sh

# 从工作流中还原 (带等待)
./scripts/restore-vllm.sh --from-workflow
```

#### patch-vllm.sh 参数

| 参数 | 说明 |
|------|------|
| `--restore` | 从备份还原 |
| `--backup-only` | 仅创建备份 |
| `--method line\|search` | 定位方法 (默认: search) |
| `--line-num N` | 插入行号 (用于 line 方法) |
| `--search-pattern PATTERN` | 搜索模式 (用于 search 方法) |

#### 注意事项

- CUDA 模式不需要补丁
- 补丁会自动备份原始文件 (`.bak`)
- 补丁位置: `site-packages/vllm/v1/worker/gpu_model_runner.py`

---

## 目录结构

```
FlagTune/
├── config.yaml                    # 用户主配置
├── scripts/
│   ├── setup-deps.sh             # 依赖安装
│   ├── patch-vllm.sh             # 应用 vLLM 补丁
│   ├── restore-vllm.sh           # 还原 vLLM 补丁
│   ├── target_ops.sh             # 目标算子列表定义
│   │
│   ├── run-workflow.sh           # 核心工作流调度器
│   ├── run-processing.sh         # 数据处理调度器
│   │
│   ├── auto-workflow.sh          # 一键自动测试 (推荐入口)
│   ├── auto-processing.sh        # 一键自动处理 (推荐入口)
│   │
│   ├── export-nsys.sh            # NSYS 结果导出
│   │
│   └── tools/                    # 工具脚本
│       ├── tool_config.yaml      # 运行时配置 (自动生成)
│       ├── benchmark_runner.py   # 基准测试执行器
│       ├── perf_analysis.py      # 性能对比分析
│       ├── gems_shape_info.py    # 算子形状信息提取
│       ├── bench_stat.py         # 基准测试统计报告
│       └── benchmark_throughput_flagos_statistics.py  # 吞吐量统计
│
├── processing/                    # 数据处理脚本目录
├── results/{model.name}/          # 测试结果
│   ├── server-logs/               # 服务器日志
│   ├── bench_optimized_log/       # Optimized 模式日志
│   ├── bench_log/                 # Benchmark 模式日志
│   ├── nsys-raw/                  # NSYS 原始数据
│   └── torch-raw/                 # Torch Profiler 数据
│
└── reports/{model.name}/          # 分析报告
    ├── bench_stat_{suffix}.md     # 基准测试统计报告
    ├── perf_analysis.md           # 性能对比报告
    ├── perf_analysis.xlsx         # Excel 格式报告
    ├── shape_analysis.md          # 形状分析报告
    └── gems_shape_info.txt        # 算子形状信息
```

---

## 工具脚本详解

### 1. setup-deps.sh - 依赖安装

```bash
# 安装所有依赖
./scripts/setup-deps.sh

# 仅检查依赖状态
./scripts/setup-deps.sh --check
```

### 2. export-nsys.sh - NSYS 结果导出

将 `.nsys-rep` 文件导出为 `.sqlite` 格式用于分析。

```bash
# 使用 tool_config.yaml 中的路径
./scripts/export-nsys.sh

# 指定目录
./scripts/export-nsys.sh --dir results/your-model/nsys-raw
```

### 3. target_ops.sh - 目标算子列表

定义批量测试的目标算子列表，可被其他脚本引用。

```bash
# 查看默认算子列表
./scripts/target_ops.sh

# 从配置文件加载
source ./scripts/target_ops.sh /path/to/config.yaml
```

**默认算子列表**: reciprocal, cat, gather, lt, le, softmax, scatter, cumsum_out, layer_norm

---

## Python 工具详解

### 1. benchmark_runner.py

核心基准测试执行器，由 `run-workflow.sh` 调用。

**功能**:
- 支持 `vllm bench serve` 和 `vllm bench throughput` 模式
- 集成 NSYS 性能分析
- 集成 Torch Profiler
- 多场景、多轮次测试
- 日志自动保存

### 2. bench_stat.py

基准测试结果统计工具。

```bash
# 分析优化模式结果
python scripts/tools/bench_stat.py -f optimized --warmup 1

# 分析完整基准结果
python scripts/tools/bench_stat.py -f full --warmup 1
```

**输出**: Markdown 格式的对比报告，包含:
- Output Token Mean Throughput 对比
- Total Token Mean Throughput 对比
- Speedup 比例 (GEMS/CUDA)

### 3. perf_analysis.py

NSYS 性能深度分析工具。

```bash
python scripts/tools/perf_analysis.py \
  --nsys_path results/your-model/nsys-raw \
  --output_path reports/your-model/
```

**输出**:
- `perf_analysis.md` - Markdown 性能对比报告
- `perf_analysis.xlsx` - Excel 格式报告
- `shape_analysis.md` - 形状分析
- `shape_analysis.xlsx` - Excel 形状分析

### 4. gems_shape_info.py

从 FlagGems 日志中提取算子形状信息。

```bash
python scripts/tools/gems_shape_info.py \
  --input results/your-model/gems-config/gems-all.txt \
  --output reports/your-model/gems_shape_info.txt
```

### 5. benchmark_throughput_flagos_statistics.py

吞吐量统计分析工具。

```bash
python scripts/tools/benchmark_throughput_flagos_statistics.py \
  --log-dir results/your-model/bench_optimized_log/
```

**输出统计信息**:
- Mean, Median, Max, Standard Deviation
- Max deviation in sigma units
- Output token throughput 和 Total token throughput

---

## 结果说明

### 输出目录结构

```
results/your-model/
├── server-logs/                           # 服务器日志
│   ├── vllm_bench_cuda_server.log
│   └── vllm_bench_gems_all_server.log
│
├── bench_optimized_log/                   # Optimized 模式日志
│   ├── vllm_bench_cuda_logs/
│   │   └── p4096d1024_run*.log
│   └── vllm_bench_gems_all_logs/
│       └── p4096d1024_run*.log
│
├── bench_log/                             # 完整 Benchmark 日志
├── nsys-raw/                              # NSYS 原始数据
│   ├── report-cuda.nsys-rep
│   ├── report-cuda.sqlite
│   ├── report-gems-all.nsys-rep
│   └── report-gems-all.sqlite
│
├── torch-raw/                             # Torch Profiler 数据
└── gems-config/                           # Gems 配置和日志
    └── gems-all.txt

reports/your-model/
├── bench_stat_optimized.md                # 基准测试统计报告
├── perf_analysis.md                       # 性能分析报告
├── perf_analysis.xlsx                     # Excel 性能报告
├── shape_analysis.md                      # 形状分析报告
└── gems_shape_info.txt                    # 算子形状信息
```

### 查看结果

```bash
# 查看基准测试日志
cat results/your-model/bench_optimized_log/vllm_bench_gems_all_logs/p4096d1024_run1.log

# 查看统计报告
cat reports/your-model/bench_stat_optimized.md

# 查看性能分析报告
cat reports/your-model/perf_analysis.md
```

---

## 会话管理

### tmux 命令

```bash
# 查看服务器状态
tmux attach -t screen

# 退出但不停止服务 (按 Ctrl+B 然后按 D)

# 停止服务
tmux send-keys -t screen:bench C-c

# 终止会话
tmux kill-session -t screen
```

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `USE_FLAGOS=1` | 启用 FlagGems |
| `USE_GEMS_MODE` | 算子模式: all, NULL, 或具体算子名 |
| `GEMS_ONCE` | 是否只启用一次 (true/false) |
| `GEMS_SAVE_PATH` | Gems 日志保存路径 |
| `GEMS_UNUSE` | 禁用的算子列表 |

---

## 常见问题

### Q: 如何测试单个算子?

```bash
./scripts/run-workflow.sh --mode gems --device 0 --optimized --gems-mode layer_norm
```

### Q: 如何批量测试多个算子?

```bash
./scripts/run-workflow.sh --mode gems --device 0 --optimized --batch
```

### Q: OOM 怎么办?

修改 `config.yaml`:
```yaml
serve:
  gpu_memory_utilization: 0.8    # 降低
  max_num_batched_tokens: 8192   # 降低
  max_num_seqs: 1024             # 降低
```

### Q: 如何查看服务器日志?

```bash
# 方法1: tmux
tmux capture-pane -t screen:bench -p -S -100

# 方法2: 日志文件
cat results/your-model/server-logs/*.log | tail -100
```

### Q: 依赖安装失败?

```bash
# 手动安装
sudo apt install jq tmux curl
pip install pyyaml

# 安装 yq
curl -fsSL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /tmp/yq
chmod +x /tmp/yq
sudo mv /tmp/yq /usr/local/bin/yq
```

### Q: 如何快速运行完整测试?

```bash
# 一键运行所有测试
./scripts/auto-workflow.sh --model your-model --device 0 --all

# 等待完成后，一键处理结果
./scripts/auto-processing.sh --model your-model --all
```

---

## 更多帮助

- 使用 `/flagtune:setup` 进行环境配置
- 使用 `/flagtune:debug` 进行问题调试
- 使用 `/flagtune:analyze` 进行结果分析
