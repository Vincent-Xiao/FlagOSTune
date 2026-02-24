#!/usr/bin/env python3
"""
Scan `vllm_bench_log` subdirectories, for each:
 1. replace LOG_DIR in vllm-plugin-FL/benchmarks/benchmark_throughput_flagos_statistics.py
 2. run that script and capture stdout
 3. convert any found table to a Markdown table and append to reports/bench-report-<date>.md

This script restores the original benchmark file content at the end.
"""
from pathlib import Path
import argparse
import re
import subprocess
import sys
from datetime import date


def find_bench_subdirs(root: Path, bench_dir: Path):
    if not bench_dir.exists():
        raise FileNotFoundError(f"{bench_dir} not found")
    subs = [p for p in bench_dir.iterdir() if p.is_dir()]
    def priority(p: Path):
        name = p.name.lower()
        if 'cuda' in name:
            return (0, name)
        if 'all' in name:
            return (1, name)
        return (2, name)
    return sorted(subs, key=priority)


def replace_log_dir(bench_file: Path, new_dir: str):
    content = bench_file.read_text(encoding="utf-8")
    pattern = re.compile(r"^LOG_DIR\s*=.*$", flags=re.M)
    new_assignment = f"LOG_DIR = '{new_dir}'"
    if pattern.search(content):
        new_content = pattern.sub(new_assignment, content)
    else:
        # If not found, append assignment at top
        new_content = new_assignment + "\n" + content
    bench_file.write_text(new_content, encoding="utf-8")


def format_bench_title(dir_name: str) -> str:
    """Convert names like 'vllm_bench_<core>_logs' to '<core> benchmark'."""
    if dir_name.startswith("vllm_bench_") and dir_name.endswith("_logs"):
        core = dir_name[len("vllm_bench_"):-len("_logs")]
        return f"{core} benchmark"
    return dir_name


def to_float(x):
    """Convert a value to float or return None for non-numeric / N/A."""
    try:
        return float(str(x).replace(',', ''))
    except Exception:
        return None


def text_table_to_markdown(text: str) -> str:
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""

    rows = []
    # If lines contain '|' treat as pipe-separated
    if any("|" in l for l in lines):
        for l in lines:
            # ignore separator lines like |----|
            if re.match(r"^\s*\|?\s*-{2,}", l):
                continue
            parts = [c.strip() for c in l.strip().strip("|").split("|")]
            rows.append(parts)
    else:
        splitter = re.compile(r"\s{2,}")
        for l in lines:
            parts = [p.strip() for p in splitter.split(l) if p.strip()]
            rows.append(parts)

    if not rows:
        return ""

    # Normalize column count
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    header = rows[0]
    body = rows[1:]

    sep = ["---"] * len(header)
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in body:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)


def parse_text_table(text: str):
    """Parse a text or pipe table into (header, rows) where rows are lists of columns."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if not lines:
        return [], []

    rows = []
    if any("|" in l for l in lines):
        for l in lines:
            if re.match(r"^\s*\|?\s*-{2,}", l):
                continue
            parts = [c.strip() for c in l.strip().strip("|").split("|")]
            rows.append(parts)
    else:
        splitter = re.compile(r"\s{2,}")
        for l in lines:
            parts = [p.strip() for p in splitter.split(l) if p.strip()]
            rows.append(parts)

    if not rows:
        return [], []

    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    header = rows[0]
    body = rows[1:]
    return header, body


def run_benchmark(script_path: Path):
    proc = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    parser = argparse.ArgumentParser(description="Process benchmark statistics")
    parser.add_argument("--f", dest="filename", type=str, default="", help="Filename suffix for input directory and output report")
    args = parser.parse_args()
    
    script_file = Path(__file__).resolve()
    repo_root = script_file.parent.parent
    bench_script = repo_root / "benchmarks" / "benchmark_throughput_flagos_statistics.py"
    if not bench_script.exists():
        print(f"Error: benchmark script not found at {bench_script}")
        sys.exit(1)

    orig_content = bench_script.read_text(encoding="utf-8")
    
    # Construct bench_dir based on filename parameter
    if args.filename:
        bench_dir = repo_root / f"vllm_bench_{args.filename}_log"
    else:
        bench_dir = repo_root / "vllm_bench_log"
    
    try:
        subdirs = find_bench_subdirs(repo_root, bench_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    report_dir = repo_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct report_file based on filename parameter
    if args.filename:
        report_file = report_dir / f"bench-{args.filename}-report-{date.today().isoformat()}.md"
    else:
        report_file = report_dir / f"bench-report-{date.today().isoformat()}.md"

    # Open in write mode so each run overwrites previous report
    # collect Out Mean and Tot Mean data per subdir
    collected = {}
    collected_tot = {}
    scenarios_order = []

    with report_file.open("w", encoding="utf-8") as rf:
        for sd in subdirs:
            display_name = format_bench_title(sd.name)
            print(f"Running benchmark for: {display_name} ({sd})")
            replace_log_dir(bench_script, str(sd))
            rc, out, err = run_benchmark(bench_script)
            if rc != 0:
                rf.write(f"\n## {display_name} - ERROR (return code {rc})\n\n")
                rf.write("```")
                rf.write(err or out)
                rf.write("```\n")
                print(f"Benchmark failed for {sd}: return code {rc}")
                continue

            md_table = text_table_to_markdown(out)
            rf.write(f"\n## {display_name}\n\n")
            if md_table:
                rf.write(md_table + "\n")
            else:
                rf.write("```")
                rf.write(out)
                rf.write("```\n")

            # parse table and collect Out Mean and Tot Mean values
            header, rows = parse_text_table(out)
            if header and rows:
                # normalize header names
                lower_header = [h.lower() for h in header]
                try:
                    scen_idx = lower_header.index('scenario')
                except ValueError:
                    scen_idx = 0
                out_mean_idx = None
                tot_mean_idx = None
                for i, h in enumerate(lower_header):
                    if 'out mean' == h or h.startswith('out mean') or h == 'out':
                        out_mean_idx = i
                    if 'tot mean' == h or h.startswith('tot mean') or h == 'tot':
                        tot_mean_idx = i

                sd_map = {}
                sd_tot_map = {}
                for r in rows:
                    scen = r[scen_idx]
                    if scen not in scenarios_order:
                        scenarios_order.append(scen)
                    out_val = r[out_mean_idx] if out_mean_idx is not None and out_mean_idx < len(r) else ''
                    tot_val = r[tot_mean_idx] if tot_mean_idx is not None and tot_mean_idx < len(r) else ''
                    if out_val and out_val.strip().lower() != 'n/a':
                        sd_map[scen] = out_val
                    if tot_val and tot_val.strip().lower() != 'n/a':
                        sd_tot_map[scen] = tot_val
                collected[sd.name] = sd_map
                collected_tot[sd.name] = sd_tot_map

    # After processing all, write combined table of Out Mean (non-N/A)
    if collected:
        # union of scenarios
        scenarios = scenarios_order
        # map subdir names to simplified labels
        def label_for(name: str) -> str:
            n = name.lower()
            if n.startswith("vllm_bench_") and n.endswith("_logs"):
                body = n[len("vllm_bench_"):-len("_logs")]
                # remove any trailing slash/underscores just in case
                return body.strip('_')
            return n

        # display label for table header
        def display_label(col_name: str) -> str:
            # when running with --f plugin, rename cuda and gems_* columns for clarity
            if args.filename == "plugin":
                if col_name == "cuda":
                    return "cuda_plugin"
                if col_name.startswith("gems_"):
                    return "gems_plugin_" + col_name[len("gems_"):]
            return col_name

        # build label -> subdir mapping from both collected and collected_tot keys
        all_sd_names = set(collected.keys()) | set(collected_tot.keys())
        label_to_sd = {}
        for sd_name in all_sd_names:
            lbl = label_for(sd_name)
            if lbl not in label_to_sd:
                label_to_sd[lbl] = sd_name

        # preferred order
        preferred = ['cuda', 'gems_all', 'gems_72', 'gems_gather', 'gems_layer_norm']
        cols = []
        # first add preferred if they exist
        for p in preferred:
            if p in label_to_sd:
                cols.append(p)
        # then others
        for l in label_to_sd.keys():
            if l not in cols:
                cols.append(l)

        # We will add speedup columns for everything that starts with 'gems_' against 'cuda'
        speedup_cols = []
        if 'cuda' in cols:
            for c in cols:
                if c != 'cuda' and c.startswith('gems_'):
                    speedup_cols.append(f"{c}/cuda")
        
        # create combined markdown with simplified column headers
        with report_file.open("a", encoding="utf-8") as rf:
            rf.write('\n## Output Token Mean Throughput （tokens/s）对比\n\n')

            # header display names
            display_cols = [display_label(c) for c in cols]
            display_speedup_cols = []
            for sc in speedup_cols:
                base, sep, rest = sc.partition('/')
                display_speedup_cols.append(display_label(base) + sep + rest)

            header = ['Scenario'] + display_cols + display_speedup_cols
            rf.write('| ' + ' | '.join(header) + ' |\n')
            rf.write('| ' + ' | '.join(['---'] * len(header)) + ' |\n')
            
            # determine indices
            col_indices = {c: i for i, c in enumerate(cols)}
            cuda_idx = col_indices.get('cuda')

            # collect rows so we can compute column-wise averages and per-row speedups
            table_rows = []
            speedup_data = {sc: [] for sc in speedup_cols} # store numeric speedups for min/max

            for s in scenarios:
                row = [s]
                values = []
                for lbl in cols:
                    sd_name = label_to_sd.get(lbl)
                    val = collected.get(sd_name, {}).get(s, 'N/A')
                    values.append(val)
                    row.append(val)

                if any((not v or v.strip().lower() == 'n/a') for v in values):
                    continue

                # Compute speedups
                for sc in speedup_cols:
                    # sc covers format "X/cuda"
                    target_col_name = sc.split('/')[0]
                    target_idx = col_indices.get(target_col_name)
                    
                    ratio_str = 'N/A'
                    if cuda_idx is not None and target_idx is not None:
                         cuda_val = to_float(values[cuda_idx])
                         target_val = to_float(values[target_idx])
                         if cuda_val is not None and cuda_val != 0 and target_val is not None:
                             ratio = target_val / cuda_val
                             ratio_str = f"{ratio:.3f}"
                             speedup_data[sc].append(ratio)
                         else:
                             speedup_data[sc].append(None)
                    else:
                        speedup_data[sc].append(None)
                    row.append(ratio_str)

                rf.write('| ' + ' | '.join(row) + ' |\n')
                table_rows.append(values)

            # compute column-wise averages and write final 'Average' row
            # build numeric table once to avoid repeated conversions
            numeric_table = [[to_float(v) for v in r] for r in table_rows]

            avg_row = ['Average']
            col_avgs = [] # Numeric averages for columns
            
            for j in range(len(cols)):
                nums = [numeric_table[i][j] for i in range(len(numeric_table)) if numeric_table[i][j] is not None]
                if nums:
                    avg_val = sum(nums) / len(nums)
                    col_avg_str = f"{avg_val:.2f}"
                    col_avgs.append(avg_val)
                else:
                    col_avg_str = 'N/A'
                    col_avgs.append(None)
                avg_row.append(col_avg_str)

            # compute average speedups from averaged columns if possible
            for sc in speedup_cols:
                target_col_name = sc.split('/')[0]
                target_idx = col_indices.get(target_col_name)
                
                ratio_avg_str = 'N/A'
                if cuda_idx is not None and target_idx is not None:
                     c = col_avgs[cuda_idx]
                     t = col_avgs[target_idx]
                     if c is not None and c != 0 and t is not None:
                         ratio_avg_str = f"{(t / c):.3f}"
                avg_row.append(ratio_avg_str)
                
            # compute Min and Max rows for numeric columns and speedups
            min_row = ['Min']
            max_row = ['Max']
            for j in range(len(cols)):
                nums = [to_float(r[j]) for r in table_rows if to_float(r[j]) is not None]
                if nums:
                    min_row.append(f"{min(nums):.2f}")
                    max_row.append(f"{max(nums):.2f}")
                else:
                    min_row.append('N/A')
                    max_row.append('N/A')

            # speedup min/max
            for sc in speedup_cols:
                vals = [v for v in speedup_data[sc] if v is not None]
                if vals:
                    min_row.append(f"{min(vals):.3f}")
                    max_row.append(f"{max(vals):.3f}")
                else:
                    min_row.append('N/A')
                    max_row.append('N/A')

            # Skip Min/Max/Average rows if filename contains "optimized"
            if "optimized" not in args.filename.lower():
                rf.write('| ' + ' | '.join(min_row) + ' |\n')
                rf.write('| ' + ' | '.join(max_row) + ' |\n')
                rf.write('| ' + ' | '.join(avg_row) + ' |\n')

            # Also write combined table for Tot Mean (if available)
            if collected_tot:
                rf.write('\n## Total Token Mean Throughput （tokens/s）对比\n\n')

                # reuse display labels for Tot Mean table
                header = ['Scenario'] + display_cols + display_speedup_cols
                rf.write('| ' + ' | '.join(header) + ' |\n')
                rf.write('| ' + ' | '.join(['---'] * len(header)) + ' |\n')
                
                table_rows = []
                speedup_data = {sc: [] for sc in speedup_cols} 

                for s in scenarios:
                    row = [s]
                    values = []
                    for lbl in cols:
                        sd_name = label_to_sd.get(lbl)
                        val = collected_tot.get(sd_name, {}).get(s, 'N/A')
                        values.append(val)
                        row.append(val)

                    if any((not v or v.strip().lower() == 'n/a') for v in values):
                        continue

                    # Compute speedups
                    for sc in speedup_cols:
                        target_col_name = sc.split('/')[0]
                        target_idx = col_indices.get(target_col_name)
                        
                        ratio_str = 'N/A'
                        if cuda_idx is not None and target_idx is not None:
                             cuda_val = to_float(values[cuda_idx])
                             target_val = to_float(values[target_idx])
                             if cuda_val is not None and cuda_val != 0 and target_val is not None:
                                 ratio = target_val / cuda_val
                                 ratio_str = f"{ratio:.3f}"
                                 speedup_data[sc].append(ratio)
                             else:
                                 speedup_data[sc].append(None)
                        else:
                            speedup_data[sc].append(None)
                        row.append(ratio_str)
                    
                    rf.write('| ' + ' | '.join(row) + ' |\n')
                    table_rows.append(values)

                # averages
                numeric_table = [[to_float(v) for v in r] for r in table_rows]
                avg_row = ['Average']
                col_avgs = []

                for j in range(len(cols)):
                    nums = [numeric_table[i][j] for i in range(len(numeric_table)) if numeric_table[i][j] is not None]
                    if nums:
                        avg_val = sum(nums) / len(nums)
                        col_avg_str = f"{avg_val:.2f}"
                        col_avgs.append(avg_val)
                    else:
                        col_avg_str = 'N/A'
                        col_avgs.append(None)
                    avg_row.append(col_avg_str)

                for sc in speedup_cols:
                    target_col_name = sc.split('/')[0]
                    target_idx = col_indices.get(target_col_name)
                    
                    ratio_avg_str = 'N/A'
                    if cuda_idx is not None and target_idx is not None:
                         c = col_avgs[cuda_idx]
                         t = col_avgs[target_idx]
                         if c is not None and c != 0 and t is not None:
                             ratio_avg_str = f"{(t / c):.3f}"
                    avg_row.append(ratio_avg_str)

                # Min/Max
                min_row = ['Min']
                max_row = ['Max']
                for j in range(len(cols)):
                    nums = [to_float(r[j]) for r in table_rows if to_float(r[j]) is not None]
                    if nums:
                        min_row.append(f"{min(nums):.2f}")
                        max_row.append(f"{max(nums):.2f}")
                    else:
                        min_row.append('N/A')
                        max_row.append('N/A')

                for sc in speedup_cols:
                    vals = [v for v in speedup_data[sc] if v is not None]
                    if vals:
                        min_row.append(f"{min(vals):.3f}")
                        max_row.append(f"{max(vals):.3f}")
                    else:
                        min_row.append('N/A')
                        max_row.append('N/A')

                # Skip Min/Max/Average rows if filename contains "optimized"
                if "optimized" not in args.filename.lower():
                    rf.write('| ' + ' | '.join(min_row) + ' |\n')
                    rf.write('| ' + ' | '.join(max_row) + ' |\n')
                    rf.write('| ' + ' | '.join(avg_row) + ' |\n')

    # restore original file content
    bench_script.write_text(orig_content, encoding="utf-8")
    print(f"Report written to: {report_file}")


if __name__ == "__main__":
    main()
