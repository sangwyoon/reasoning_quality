#!/usr/bin/env python3
import argparse, json, glob, os, sys
from collections import OrderedDict

DATASETS = ("gsm8k", "math_t", "omni", "gpqa", "mmlu", "agie", "mmlupro")

def uids(path_glob):
    ids = set()
    for fp in glob.glob(path_glob):
        for line in open(fp):
            try:
                ids.add(json.loads(line)["question_id"])
            except Exception:
                pass
    return ids

parser = argparse.ArgumentParser(description="List questions missing from ablation / KL outputs.")
parser.add_argument("--base", default="results", help="directory holding the runner outputs")
BASE = parser.parse_args().base

manifest = OrderedDict()
summary = []
cot_dirs = []
max_n = {}
for cot_dir in sorted(glob.glob(f"{BASE}/exp1_cot_*")):
    name = os.path.basename(cot_dir)[len("exp1_cot_"):]
    ds = next((k for k in DATASETS if name.startswith(k + "_")), None)
    if ds is None:
        continue
    n = len(uids(f"{cot_dir}/*.jsonl"))
    cot_dirs.append((cot_dir, name, ds))
    max_n[ds] = max(max_n.get(ds, 0), n)
if not cot_dirs:
    sys.exit(f"no exp1_cot_* outputs found under {BASE!r}")
for cot_dir, name, ds in cot_dirs:
    tag = name[len(ds) + 1:]
    cot_ids = uids(f"{cot_dir}/*.jsonl")
    if len(cot_ids) < max_n[ds] * 0.99:
        continue
    for phase, dpre in (("abl", "abl_sent"), ("kl", "kl_mask10")):
        d = f"{BASE}/{dpre}_{ds}_{tag}"
        if not os.path.isdir(d):
            continue
        present = uids(f"{d}/*.jsonl")
        excluded = sorted(cot_ids - present, key=lambda q: (len(q), q))
        if excluded:
            manifest[f"{phase}/{ds}/{tag}"] = {
                "n_expected": len(cot_ids), "n_present": len(present),
                "excluded": excluded,
            }
            summary.append((phase, ds, tag, len(excluded)))

out = f"{BASE}/excluded_questions.json"
json.dump(manifest, open(out, "w"), ensure_ascii=False, indent=1)
print(f"{out} written: {len(manifest)} combos with exclusions")
for phase, ds, tag, n in sorted(summary, key=lambda x: -x[3])[:15]:
    print(f"  {phase:3s} {ds:8s} {tag:34s} {n} items")
total = sum(n for *_, n in summary)
print(f"Total excluded records: {total}")
