#!/usr/bin/env python3
"""
Benchmark throughput statistics for vLLM.

This script analyzes benchmark logs and outputs statistics including
mean, median, max, standard deviation, and max deviation in sigma units.
"""
import argparse
import re
import os
import statistics
from pathlib import Path

SCENARIOS = [
    # optimized scenarios
    "p1d1024",
    "p65536d1024",
    # legacy scenarios (kept for compatibility)
    "p128d128",
    "p6144d128",
    "p30720d128",
    "p128d6144",
    "p6144d6144",
    "p30720d6144",
    "p4096d2048",
    "p6144d1024",
    "p4096d1024",
    "p2048d1024",
    "p1024d1024"
]


def extract_throughputs(log_path: str):
    output_throughput = None
    total_throughput = None
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return None, None

    out_match = re.search(r'Output token throughput \(tok/s\):\s*([\d.]+)', content)
    tot_match = re.search(r'Total token throughput \(tok/s\):\s*([\d.]+)', content)

    if out_match:
        output_throughput = float(out_match.group(1))
    if tot_match:
        total_throughput = float(tot_match.group(1))

    return output_throughput, total_throughput


def compute_extended_stats(values):
    valid_vals = [v for v in values if v is not None]
    if not valid_vals:
        return ("N/A", "N/A", "N/A", "N/A", "N/A")

    mean_val = statistics.mean(valid_vals)
    median_val = statistics.median(valid_vals)
    max_val = max(valid_vals)

    stdev_val = 0.0 if len(valid_vals) == 1 else statistics.stdev(valid_vals)

    max_abs_dev = max(abs(x - mean_val) for x in valid_vals)
    if stdev_val > 0:
        sigma_str = f"{max_abs_dev / stdev_val:.2f} sigma"
    else:
        sigma_str = "0.00 sigma" if max_abs_dev == 0 else "N/A"

    return (
        f"{mean_val:.2f}",
        f"{median_val:.2f}",
        f"{max_val:.2f}",
        f"{stdev_val:.2f}",
        sigma_str
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark throughput statistics")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory containing benchmark log files"
    )
    args = parser.parse_args()

    # Use command-line argument if provided, otherwise use default
    if args.log_dir:
        LOG_DIR = args.log_dir
    else:
        # Default path for backward compatibility
        LOG_DIR = '/root/minicpm-vllm-xh/vllm_bench_log/vllm_bench_cuda_logs'

    # Verify directory exists
    if not os.path.isdir(LOG_DIR):
        print(f"Error: Log directory not found: {LOG_DIR}")
        return 1

    # aligned output
    scene_width = 14
    num_width = 10
    sigma_width = 12

    # headers
    header = (
        f"{'Scenario':<{scene_width}} | "
        f"{'Out Mean':>{num_width}} | {'Out Med':>{num_width}} | {'Out Max':>{num_width}} | {'Out sigma':>{num_width}} | {'Out Dev':>{sigma_width}} | "
        f"{'Tot Mean':>{num_width}} | {'Tot Med':>{num_width}} | {'Tot Max':>{num_width}} | {'Tot sigma':>{num_width}} | {'Tot Dev':>{sigma_width}}"
    )
    print(header)
    print("-" * len(header))

    # only run2, run3, run4, ignore run1
    RUN_IDS = [2, 3, 4]

    for scene in SCENARIOS:
        output_vals = []
        total_vals = []

        for run_id in RUN_IDS:
            log_file = os.path.join(LOG_DIR, f"{scene}_run{run_id}.log")
            out_tok_s, total_tok_s = extract_throughputs(log_file)
            output_vals.append(out_tok_s)
            total_vals.append(total_tok_s)

        out_stats = compute_extended_stats(output_vals)
        tot_stats = compute_extended_stats(total_vals)

        line = (
            f"{scene:<{scene_width}} | "
            f"{out_stats[0]:>{num_width}} | {out_stats[1]:>{num_width}} | {out_stats[2]:>{num_width}} | {out_stats[3]:>{num_width}} | {out_stats[4]:>{sigma_width}} | "
            f"{tot_stats[0]:>{num_width}} | {tot_stats[1]:>{num_width}} | {tot_stats[2]:>{num_width}} | {tot_stats[3]:>{num_width}} | {tot_stats[4]:>{sigma_width}}"
        )
        print(line)

    return 0


if __name__ == "__main__":
    exit(main())
