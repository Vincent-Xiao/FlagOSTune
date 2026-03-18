#!/usr/bin/env bash
set -euo pipefail

# benchmark
## cuda optimized场景bencahmark测试
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode cuda --scenario optimized
## gems optimized场景bencahmark测试
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode gems --scenario optimized
## optimized场景性能报告
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow bench

# torch profling
## cuda optimized场景torch profling
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode cuda --torch
## gems optimized场景torch profling
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode gems --torch
## cuda optmized场景torch profling性能报告
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode cuda
## gems optmized场景torch profling性能报告
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode gems
## cuda,gems optmized场景torch profling性能对比，--mode compare(默认)会运行两次torch profling，分别是cuda和gems场景，不用单独执行cuda和gems场景的torch profling报告
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode compare

# gems算子导出shape
## gems shape场景算子shape导出
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --scenario shape
## gems shape场景算子shape 解析
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow shape