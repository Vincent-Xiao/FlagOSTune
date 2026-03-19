# FlagTune 文档

FlagTune 是一个面向 vLLM + CUDA/FlagGems 的性能测试与分析项目，用于统一管理模型配置、执行 benchmark 测试，并生成 Torch Profiler 分析报告。项目目标是把模型评测流程标准化，便于对比 CUDA 与 FlagGems 在不同模型和场景下的性能表现。

## 1. 模型 config 配置

模型配置文件命名规则：

```bash
config.yaml.<模型名>
```

例如当前仓库内已有：

- `config.yaml.Qwen3.5-35B-A3B`
- `config.yaml.Qwen3.5-397B-A17B`
- `config.yaml.Deepseek-3.2`
- `config.yaml.glm-5-fp8`
- `config.yaml.kimi-K2.5`

新模型建议从模板复制：

```bash
cp config.yaml.template config.yaml.<模型名>
```

一个典型配置如下：

```yaml
model:
  path: /models/Qwen3.5-35B-A3B
  name: Qwen3.5-35B-A3B
  tensor_parallel_size: 1
  tokenizer_path: null

serve:
  gpu_memory_utilization: 0.85
  max_num_batched_tokens: 16384
  max_num_seqs: 2048
  trust_remote_code: true
  reasoning_parser: qwen3
  load_format: auto
  extra_args: ""

benchmark:
  host: 127.0.0.1
  port_base: 2345
  num_runs: 5
  scenarios:
    optimized:
      - name: p32768d1024
        input_len: 32768
        output_len: 1024
        concurrency: 8
    full:
      - name: p128d128
        input_len: 128
        output_len: 128
        concurrency: 100
    shape:
      - name: p1024d1024
        input_len: 1024
        output_len: 1024
        concurrency: 100

target_ops:
  - reciprocal
  - cat
  - gather
  - lt
  - le
  - softmax
  - scatter
  - cumsum_out
  - layer_norm

paths:
  results: results
  reports: reports
  use_model_name: true
```

重点字段：

- `model.path`：模型目录
- `model.name`：vllm服务名
- `model.tensor_parallel_size`：tp并行数
- `benchmark.num_runs`：每个场景重复次数
- `benchmark.scenarios.optimized`：快速测试场景
- `benchmark.scenarios.full`：完整 benchmark 场景
- `benchmark.scenarios.shape`：flaggems 算子shape导出场景
- `paths.results` / `paths.reports`：结果与报告目录

---

## 2. Benchmark 测试

### 2.1 运行测试

使用 `auto-workflow.sh` 运行 benchmark：

```bash
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode cuda --scenario optimized
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode gems --scenario optimized
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --scenario optimized
```

常用参数：

- `--model`：选择 `config.yaml.<模型名>`
- `--mode cuda|gems|all`：运行目标，默认 `all`(即同时运行 CUDA 和 FlagGems)
- `--device`：GPU 编号
- `--scenario optimized|full|shape`：测试场景
- `--gems-mode`：指定 FlagGems 模式，默认 `all`

### 2.2 结果目录

运行后数据默认保存在：

- `results/<model>/bench_optimized_log/...`
- `results/<model>/bench_log/...`

### 2.3 生成 benchmark 报告

```bash
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow bench
```

默认会处理 optimized benchmark 结果，并在下面生成报告：

```bash
reports/<model>/bench-optimized-report-<date>.md
```

---

## 3. Torch Profiler 测试与报告

### 3.1 运行 Torch Profiler 测试

推荐分别采集 CUDA 和 FlagGems 的 profiler 原始数据：

```bash
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode cuda --torch
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode gems --torch
```

说明：

- `--torch` 模式下会强制使用 `benchmark.num_runs=2`
- 如需一次性同时采集cuda和flagems双侧数据，也可以直接执行：
  - `./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --torch`
- profiler 原始数据默认输出到：
  - `results/<model>/torch-raw/report-cuda`
  - `results/<model>/torch-raw/report-gems-all`

### 3.2 生成 Torch Profiler 报告

处理单侧结果：

```bash
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode cuda
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode gems
```

执行对比分析：不用执行单侧处理

```bash
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode compare
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode compare --rank 0
```

说明：

- `--mode compare` 只对已有的 CUDA / FlagGems profiler 结果做对比分析
- `--rank` 默认为 `0`，也可指定 `all`
- 若直接执行 `./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch`，默认等价于 `--mode compare`

生成文件：

- `reports/<model>/perf_analysis_torch.md`
- `reports/<model>/perf_analysis_torch.xlsx`
- `reports/perf_summary_torch.md`
- `reports/perf_summary_torch.xlsx`

如需汇总多个模型的 torch profiling 报告：

```bash
python scripts/tools/perf_summary_torch.py
```

报告内容包括：

- CUDA kernel 排序结果
- FlagGems kernel 排序结果
- CUDA / FlagGems 对比表
- 按 `rank` 维度的 profiler 统计

---

## 4. Gems shape 导出

如果需要导出 FlagGems 的 shape 信息，先运行 shape 场景：

```bash
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --scenario shape
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --scenario shape --gems-mode mm
```

shape 原始导出文件默认保存在：

- `results/<model>/gems-config-shape/gems-all.txt`
- `results/<model>/gems-config-shape/marker.txt`

再执行 shape 处理：

```bash
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow shape
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow shape --gems-mode mm #只导出mm算子的shape
```

处理后会在下面生成按场景拆分的 shape 文件：

```bash
reports/<model>/shape/*.txt
```

例如：

- `reports/Qwen3.5-35B-A3B/shape/Qwen3.5-35B-A3B-p1024d1024.txt`
- `reports/Qwen3.5-35B-A3B/shape/Qwen3.5-35B-A3B-p32768d1024.txt`

---

## 5. function_test.sh 说明

[`function_test.sh`](function_test.sh) 是一个简单的功能验证脚本，用来串联本仓库常用命令，方便快速执行以下流程：

- optimized benchmark 测试与报告生成
- torch profiling 采集、单侧分析、对比分析
- 多模型 torch profiling 汇总
- FlagGems shape 导出与解析

使用时可直接参考脚本中的命令顺序，按需手动执行，或将其作为日常回归测试的参考清单。
