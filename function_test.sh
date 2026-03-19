#!/usr/bin/env bash

# ## cuda optimized场景bencahmark测试
# ./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode cuda --scenario optimized
# ## gems optimized场景bencahmark测试
# ./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode gems --scenario optimized
# ## optimized场景性能报告
# ./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow bench

# ## cuda optimized场景torch profling
# ./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode cuda --torch
## gems optimized场景torch profling
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode gems --torch
## cuda optmized场景torch profling性能报告
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode cuda
## gems optmized场景torch profling性能报告
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode gems
## cuda,gems optimized场景torch profiling性能对比，--mode compare只对已有的cuda和gems profiler结果做对比分析
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode compare
## 汇总所有模型的torch profiling性能报告
python scripts/tools/perf_summary_torch.py

## gems shape场景算子shape导出
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --scenario shape
## gems shape场景算子shape 解析
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow shape
