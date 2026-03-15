#!/usr/bin/env python3
"""
benchmark_runner.py - FlagTune 基准测试执行器

从 tool_config.yaml 读取所有配置，执行 vLLM 基准测试。
支持多种场景和运行模式。

注意: 所有配置由 run-workflow.sh 从 config.yaml 复制到 tool_config.yaml
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# 尝试导入 torch
try:
    import torch
    import contextlib
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    contextlib = None


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent


def load_config() -> dict:
    """加载配置 (仅从 tool_config.yaml)"""
    config_path = get_project_root() / "scripts" / "tools" / "tool_config.yaml"

    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_scenarios(config: dict) -> list:
    """获取测试场景"""
    current_run = config.get('current_run', {})
    scenario_type = current_run.get('scenario_type')

    # 兼容旧配置：如果没有 scenario_type，则回退到 optimized 布尔值
    if not scenario_type:
        optimized = current_run.get('optimized', False)
        scenario_type = 'optimized' if optimized else 'full'

    scenarios = config.get('benchmark', {}).get('scenarios', {}).get(scenario_type, [])

    # 默认场景
    if not scenarios:
        scenarios = [
            {'name': 'p4096d1024', 'input_len': 4096, 'output_len': 1024, 'concurrency': 64},
        ]

    return scenarios


def build_benchmark_command(scenario: dict, config: dict, is_last_run: bool = False) -> list:
    """构建基准测试命令"""
    current_run = config.get('current_run', {})
    nsys_profile = current_run.get('nsys_profile', False)
    torch_profile = current_run.get('torch_profile', False)

    host = config.get('benchmark', {}).get('host', '127.0.0.1')
    port = current_run.get('port', 2345)
    model_path = config.get('model', {}).get('path', '')
    model_name = config.get('model', {}).get('name', 'model')
    tokenizer_path = config.get('model', {}).get('tokenizer_path') or model_path
    tensor_parallel_size = config.get('model', {}).get('tensor_parallel_size', 8)

    input_len = scenario.get('input_len', 1024)
    output_len = scenario.get('output_len', 1024)
    concurrency = scenario.get('concurrency', 100)

    if torch_profile:
        # Torch profiling 使用 throughput 模式
        torch_report_dir = Path(
            config.get('paths', {}).get('torch_output_dir',
                                        'results/torch-raw'))
        mode = current_run.get('mode', 'cuda')
        gems_mode = current_run.get('gems', {}).get('mode', '')
        gems_suffix = f"-{gems_mode}" if mode == 'gems' else ""
        torch_report_dir = torch_report_dir / f"report-{mode}{gems_suffix}"
        torch_report_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            'vllm',
            'bench',
            'throughput',
            '--tensor-parallel-size',
            str(tensor_parallel_size),
            '--model',
            model_path,
            '--tokenizer',
            tokenizer_path,
            '--dataset-name',
            'random',
            '--trust-remote-code',
            '--input-len',
            str(input_len),
            '--output-len',
            str(output_len),
            '--num-prompts',
            str(concurrency),
        ]
        if is_last_run:
            cmd += [
                "--profile",
                "--profiler-config",
                f'{{"profiler": "torch","torch_profiler_dir":"{torch_report_dir}", "torch_profiler_with_stack": true, "torch_profiler_with_flops": true, "torch_profiler_use_gzip": false, "torch_profiler_dump_cuda_time_total": true, "torch_profiler_record_shapes": true, "torch_profiler_with_memory": false, "ignore_frontend": true, "delay_iterations": 2, "max_iterations": 0}}',
            ]
    else:
        # 普通模式使用 serve 模式
        cmd = [
            'vllm', 'bench', 'serve',
            '--host', host,
            '--port', str(port),
            '--backend', 'openai-chat',
            '--model', model_name,
            '--tokenizer', tokenizer_path,
            '--dataset-name', 'random',
            '--endpoint', '/v1/chat/completions',
            '--ignore-eos',
            '--trust-remote-code',
            '--random-input-len', str(input_len),
            '--random-output-len', str(output_len),
            '--num-prompts', str(concurrency),
            '--max-concurrency', str(concurrency),
        ]
        if is_last_run:
            cmd.append('--profile')

    return cmd


def run_benchmark(scenario: dict, run_id: int, config: dict):
    """运行单个基准测试"""
    name = scenario.get('name', 'unknown')
    input_len = scenario.get('input_len', 1024)
    output_len = scenario.get('output_len', 1024)
    concurrency = scenario.get('concurrency', 100)

    current_run = config.get('current_run', {})
    nsys_profile = current_run.get('nsys_profile', False)
    torch_profile = current_run.get('torch_profile', False)
    num_runs = config.get('benchmark', {}).get('num_runs', 4)

    # 判断是否为最后一轮
    is_last_run = (run_id == num_runs)
    # 获取日志目录
    log_dir = Path(config.get('paths', {}).get('log_dir', 'results/bench-logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_run{run_id}.log"

    cmd = build_benchmark_command(scenario, config, is_last_run)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Starting scenario: {name} (Run {run_id})")
    print(f"    Input: {input_len}, Output: {output_len}, Concurrency: {concurrency}")
    print(f"    Logging to: {log_file}")
    print(f"    Command: {' '.join(cmd)}\n")

    if HAS_TORCH and torch_profile and is_last_run:
        print(f"Starting torch profiling in vllm subprocess")

    # 执行命令
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    # 停止 profiler
    if HAS_TORCH and torch_profile and is_last_run:
        print(f"vllm subprocess profiling complete. ")

    status = "Success" if result.returncode == 0 else "Failed"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {status}: {name} Run {run_id} (exit code: {result.returncode})\n")


def main():
    print("=" * 60)
    print("FlagTune Benchmark Runner")
    print("=" * 60)

    # 加载配置 (仅从 tool_config.yaml)
    config = load_config()

    print(f"Mode: {config.get('current_run', {}).get('mode', 'cuda')}")
    print(f"Device: {config.get('current_run', {}).get('device', 0)}")
    print(f"Port: {config.get('current_run', {}).get('port', 2345)}")
    print(f"Model: {config.get('model', {}).get('name', 'unknown')}")

    # 获取场景
    scenarios = get_scenarios(config)
    num_runs = config.get('benchmark', {}).get('num_runs', 4)

    print(f"\nStarting vLLM benchmark suite for {len(scenarios)} scenarios, each repeated {num_runs} times...\n")

    # 运行所有场景
    for scenario in scenarios:
        for run_id in range(1, num_runs + 1):
            run_benchmark(scenario, run_id, config)

    print("=" * 60)
    print("All scenarios and runs completed.")
    log_dir = config.get('paths', {}).get('log_dir', 'results/bench-logs')
    print(f"Logs saved in: {log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
