"""Re-verify Real-FR solver results with the PBEBench verifier under each run's cascade budget k.
Real-FR has identity tasks (daughter form == proto form) whose correct answer is the empty
program; the verifier cannot parse "[]", so those are checked directly. Writes metrics/reverified_realfr.json."""
import glob, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from rewards.pbebench import reward  # noqa: E402
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
tasks = jl(f"{ROOT}/data/real_forward_reconstruction.jsonl")


def as_text(a):
    if isinstance(a, list) and a and isinstance(a[0], (list, tuple)):
        a = [f"replace({x!r}, {y!r})" for x, y in a]
    return a if isinstance(a, str) else json.dumps(a)


def ok(r, k):
    a, t = r.get("answer"), tasks[int(r["task_index"])]
    if isinstance(a, list) and len(a) == 0:
        return t["inputs"] == t["outputs"]
    return bool(a) and reward(as_text(a), True, t, max_programs=k)["value"] >= 1.0


rep = {}
for f in sorted(glob.glob(f"{ROOT}/evals/solver_results/real_fr/*/real_k*.jsonl")):
    k = int(re.search(r"real_k(\d+)", f).group(1)); R = jl(f); n = len(tasks)
    s = sum(float(r.get("best_reward") or 0) >= 1.0 for r in R); v = sum(ok(r, k) for r in R)
    rel = os.path.relpath(f, ROOT); rep[rel] = {"k": k, "self_reported_acc": round(100 * s / n, 2), "verified_acc": round(100 * v / n, 2)}
    print(f"{rel:70s} k={k:3d} self {100*s/n:5.1f} verified {100*v/n:5.1f}")
json.dump(rep, open(f"{ROOT}/metrics/reverified_realfr.json", "w"), indent=1)
