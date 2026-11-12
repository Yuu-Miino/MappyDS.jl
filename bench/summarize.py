"""Build a Markdown table from the two benchmark JSON files (Python, Julia)."""

import json
import sys

py = json.load(open(sys.argv[1]))
jl = json.load(open(sys.argv[2]))
lines = [
    "| operation | n | Python (mappy-ds) ms | Julia (MappyDS.jl) ms | Julia first call ms | speed-up |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for name in py:
    p = py[name]["per_call_ms"]
    j = jl.get(name, {}).get("per_call_ms")
    f = jl.get(name, {}).get("first_call_ms")
    if j is None:
        lines.append(f"| `{name}` | {py[name]['n']} | {p:.4f} | (missing) | | |")
        continue
    lines.append(f"| `{name}` | {py[name]['n']} | {p:.4f} | {j:.4f} | {f:.0f} | ×{p / j:.1f} |")
table = "\n".join(lines)
print(table)
if len(sys.argv) > 3:
    with open(sys.argv[3], "a") as fh:
        fh.write("## Python vs Julia per-operation timings\n\n")
        fh.write("Same runner, same systems and tolerances (rtol 1e-6, atol 1e-9; RK45 vs Tsit5). ")
        fh.write("Per-call time is the median of 3 batches after a warm-up call; the Julia first call includes compilation.\n\n")
        fh.write(table + "\n")
