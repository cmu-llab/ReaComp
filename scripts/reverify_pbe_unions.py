"""Re-verify PBEBench solver/union/hybrid accuracy with the official verifier under the dataset's
program budget (Lite 5, Hard 20), instead of trusting each solver's self-reported success.
Selection rule for unions is unchanged: highest verified reward, ties by lowest complexity,
then source order. Writes metrics/reverified_pbe.json.
"""
import glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from rewards.pbebench import reward  # noqa: E402

jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
T = {"lite": jl(f"{ROOT}/data/pbebench/lite_tasks_full_og.jsonl"), "hard": jl(f"{ROOT}/data/pbebench/tasks_full_og.jsonl")}
BUDGET = {"lite": 5, "hard": 20}
ES = f"{ROOT}/evals/solver_results"
# The six QO PBEBench solvers that make up "All Symbolic" together with the CC solver.
QO_RUNS = ["Fri_Apr_24_200_AM",
           "Sat_Apr_25_1104_PM_DEMOS_PBEBENCH_seed_42_48_examples_with_CoT",
           "Sat_Apr_25_819_PM_DEMOS_PBEBENCH_seed_42_100_examples",
           "Sun_Apr_26_120_AM_DEMOS_PBEBENCH_seed_42_12_examples_with_CoT",
           "Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT",
           "Sun_Apr_26_440_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT"]


def as_text(a):
    if a is None:
        return None
    if isinstance(a, list) and a and isinstance(a[0], (list, tuple)):  # [[A, B], ...] pairs
        a = [f"replace({x!r}, {y!r})" for x, y in a]
    return a if isinstance(a, str) else json.dumps(a)


def complexity(a):
    import re
    return sum(len(x) + len(y) for x, y in re.findall(r"replace\('((?:[^'\\]|\\.)*)',\s*'((?:[^'\\]|\\.)*)'\)", a or "")) or 10**9


def verified(path, ds):
    out = {}
    for r in jl(path):
        i = int(r["task_index"]); a = as_text(r.get("answer"))
        v = reward(a, True, T[ds][i], max_programs=BUDGET[ds])["value"] if a else 0.0
        out[i] = (v, a, float(r.get("best_reward") or 0) >= 1.0)
    return out


def union(members):
    n = max(len(m) for m in members); res = {}
    for i in range(n):
        c = [m[i] for m in members if i in m]
        best = max(x[0] for x in c)
        res[i] = min([x for x in c if x[0] >= best - 1e-9], key=lambda x: complexity(x[1]))
    return res


acc = lambda d, n: round(100 * sum(v[0] >= 1.0 for v in d.values()) / n, 2)
self_acc = lambda d, n: round(100 * sum(v[2] for v in d.values()) / n, 2)
report = {}
for ds in ("lite", "hard"):
    n = len(T[ds])
    cc = verified(f"{ES}/claude_code/Thu_Apr_23_807_PM/{ds}.jsonl", ds)
    qo_runs = {r: verified(f"{ES}/qwen3.6_35b_a3b/{r}/{ds}.jsonl", ds) for r in QO_RUNS}
    rows = {"CC Solver": cc}
    for k, v in qo_runs.items():
        rows[f"QO {k}"] = v
    q2 = qo_runs["Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT"]
    rows["CC + QO"] = union([cc, q2])
    rows["All Symbolic"] = union([cc] + list(qo_runs.values()))
    excl3 = [v for k, v in qo_runs.items() if not k.startswith("Sun_Apr_26_440_PM")]
    rows["All Symbolic (excl. QO run 3)"] = union([cc] + excl3)
    bok_path = f"{ROOT}/outputs/{'lite_tasks_full_og_best_of_k_stripped' if ds == 'lite' else 'hard_bok_converted'}.jsonl"
    bok = verified(bok_path, ds); rows["BoK"] = bok
    for name in ("CC + QO", "All Symbolic", "All Symbolic (excl. QO run 3)"):
        sym = rows[name]
        rows[f"BoK + {name}"] = {i: (1.0, None, True) if sym[i][0] >= 1.0 else bok.get(i, (0.0, None, False)) for i in range(n)}
    if ds == "lite":
        df = verified(f"{ROOT}/outputs/lite_tasks_full_og_direct_feedback_stripped.jsonl", ds); rows["DF"] = df
        for name in ("CC + QO", "All Symbolic", "All Symbolic (excl. QO run 3)"):
            sym = rows[name]
            rows[f"DF + {name}"] = {i: (1.0, None, True) if sym[i][0] >= 1.0 else df.get(i, (0.0, None, False)) for i in range(n)}
    report[ds] = {}
    print(f"\n== {ds} (n={n}, budget {BUDGET[ds]})   self-reported  ->  verified")
    for k, d in rows.items():
        report[ds][k] = {"self_reported_acc": self_acc(d, n), "verified_acc": acc(d, n),
                         "verified_mean_reward": round(sum(v[0] for v in d.values()) / n, 4)}
        flag = "" if abs(report[ds][k]["self_reported_acc"] - report[ds][k]["verified_acc"]) < 0.05 else "   <-- differs"
        print(f"  {k:72s} {report[ds][k]['self_reported_acc']:6.2f} -> {report[ds][k]['verified_acc']:6.2f}{flag}")
paper_files = {"lite": ["lite_ensemble_all_solvers", "lite_union_solvers", "lite_effi_bok_all_solvers", "lite_effi_df_all_solvers"],
               "hard": ["hard_union_all_solvers", "hard_union_bok_all_solvers", "hard_union_cc_qwen_run2", "hard_union_bok_cc_qwen_run2"]}
print("\n== union/hybrid files used for the paper tables, re-verified as stored")
for ds, names in paper_files.items():
    for nm in names:
        d = verified(f"{ROOT}/outputs/{nm}.jsonl", ds); n = len(T[ds])
        report.setdefault("paper_files", {})[nm] = {"self_reported_acc": self_acc(d, n), "verified_acc": acc(d, n)}
        print(f"  {nm:40s} {self_acc(d, n):6.2f} -> {acc(d, n):6.2f}")
json.dump(report, open(f"{ROOT}/metrics/reverified_pbe.json", "w"), indent=1)
