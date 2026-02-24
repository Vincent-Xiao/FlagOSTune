
import re
import os
import sys

# Determine project root relative to this script
# this script is at <root>/processing/import_gems_op.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

init_file = os.path.join(project_root, "FlagGems/src/flag_gems/__init__.py")
server_file = os.path.join(project_root, "vllm-minicpm/vllm/v1/worker/gpu_model_runner.py")

if not os.path.exists(init_file):
    print(f"Error: {init_file} not found")
    sys.exit(1)

if not os.path.exists(server_file):
    print(f"Error: {server_file} not found")
    sys.exit(1)

import ast

print(f"Reading {init_file}...")
with open(init_file, "r") as f:
    content = f.read()

# Use AST to parse the python file and extract registrar arguments
# This handles multiline tuples and comments correctly
ops = set()
try:
    tree = ast.parse(content)
    for node in ast.walk(tree):
        # Look for _FULL_CONFIG assignment
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_FULL_CONFIG":
                    if isinstance(node.value, ast.Tuple):
                        for elt in node.value.elts:
                             if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                                # format is ("name", function_name, [condition])
                                # we want function_name which is the second element
                                func_node = elt.elts[1]
                                if isinstance(func_node, ast.Name):
                                    ops.add(func_node.id)

        # Keep the old logic just in case, or remove it if I am sure.
        # But wait, the previous code was specifically looking for registrar call.
        # If I keep both, it might be safer if the style reverts.
        # However, checking for _FULL_CONFIG is what matches the current file.
        
        # We are looking for the call: registrar(...) inside the `enable` function usually, 
        # or globally. In the provided file it is inside `enable`.
        # checks for `registrar((...))`
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'registrar':
            if node.args and isinstance(node.args[0], ast.Tuple):
                for elt in node.args[0].elts:
                    if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                        # format is ("name", function_name, [condition])
                        # we want function_name which is the second element
                        func_node = elt.elts[1]
                        if isinstance(func_node, ast.Name):
                            ops.add(func_node.id)
except Exception as e:
    print(f"Error parsing {init_file} with AST: {e}")
    sys.exit(1)

# Deduplicate and sort
ops = sorted(list(ops))
print(f"Found {len(ops)} operators.")

print(f"Reading {server_file}...")
with open(server_file, "r") as f:
    server_content = f.read()

# Find "FlagGemsList=[" in the file
# regex search relative to the start of file
# pattern matches: FlagGemsList followed by optional whitespace and = and optional whitespace and [
list_match = re.search(r"FlagGemsList\s*=\s*\[", server_content)

if not list_match:
    print("Error: FlagGemsList=[... not found in server-gems.py")
    sys.exit(1)

# Calculate start index of the list content (immediately after "[")
start_list_content = list_match.end()

# 3. Find the matching closing bracket "]"
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

# 4. Construct new content
# Join matching ops with quotes
new_ops_str = ", ".join([f"\"{op}\"" for op in ops])

before = server_content[:start_list_content]
after = server_content[end_list_content:]

# Result: ... unused=[ <new_ops_str> ] ...
new_file_content = before + new_ops_str + after

print("Updating server-gems.py...")
with open(server_file, "w") as f:
    f.write(new_file_content)

print("Done.")
