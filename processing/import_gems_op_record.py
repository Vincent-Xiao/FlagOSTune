import re
import os
import sys

# Determine project root relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

gems_file = os.path.join(project_root, "./gems-config/gems-all.txt")
server_file = os.path.join(project_root, "/data/vllm/vllm/v1/worker/gpu_model_runner.py")

if not os.path.exists(gems_file):
    print(f"Error: {gems_file} not found")
    sys.exit(1)

if not os.path.exists(server_file):
    print(f"Error: {server_file} not found")
    sys.exit(1)

print(f"Reading {gems_file}...")
with open(gems_file, "r") as f:
    content = f.readlines()

ops = set()
# Pattern to match: flag_gems.ops.<module>.<func_name>
# Example: [DEBUG] flag_gems.ops.fill.fill_scalar_: GEMS FILL_SCALAR_
pattern = re.compile(r"flag_gems\.ops\.[a-zA-Z0-9_]+\.([a-zA-Z0-9_]+)")

for line in content:
    match = pattern.search(line)
    if match:
        op_name = match.group(1)
        ops.add(op_name)

# Deduplicate and sort
ops = sorted(list(ops))
print(f"Found {len(ops)} operators.")

print(f"Reading {server_file}...")
with open(server_file, "r") as f:
    server_content = f.read()

# Find "FlagGemsListRecord=[" in the file
# regex search relative to the start of file
list_match = re.search(r"FlagGemsListRecord\s*=\s*\[", server_content)

if not list_match:
    print("Error: FlagGemsListRecord=[... not found in server-gems.py")
    sys.exit(1)

# Calculate start index of the list content (immediately after "[")
start_list_content = list_match.end()

# Find the matching closing bracket "]"
# Scan forward from start_list_content handling nested brackets and quotes
idx = start_list_content
depth = 1
in_quote = False
quote_char = None
escape = False

while idx < len(server_content):
    char = server_content[idx]
    
    if in_quote:
        if escape:
            escape = False
        elif char == "\\":
            escape = True
        elif char == quote_char:
            in_quote = False
    else:
        if char == "\"" or char == "'":
            in_quote = True
            quote_char = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                break
    idx += 1

if depth != 0:
    print("Error: Could not find matching closing bracket for unused list")
    sys.exit(1)

end_list_content = idx # This is the index of the closing "]"

# Construct new content
# Join matching ops with quotes
new_ops_str = ", ".join([f"\"{op}\"" for op in ops])

before = server_content[:start_list_content]
after = server_content[end_list_content:]

# Result: ... FlagGemsListRecord=[ <new_ops_str> ] ...
new_file_content = before + new_ops_str + after

print("Updating server file...")
with open(server_file, "w") as f:
    f.write(new_file_content)

print("Done.")
