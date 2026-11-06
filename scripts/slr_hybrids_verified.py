"""SLR-Bench hybrids rebuilt on verified solver results: use the solver's rule only if the official
verifier accepts it, else fall back to the LLM answer (BoK / DF were run on all tasks).
Seeded DF only exists for tasks the CC solver self-reported as failed; tasks with a false CC claim
have no seeded run and are counted as unsolved (lower bound). Writes metrics/slr_hybrids_verified.json."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from slr_verify_robust import Judge, jl  # noqa: E402

if __name__ == "__main__":
    J = Judge(); tasks = jl(f"{ROOT}/data/slr_bench/v1_All_full.jsonl")
    tier = {i: t["curriculum tier"] for i, t in enumerate(tasks)}
    idx = lambda p: {int(r["task_index"]): r for r in jl(f"{ROOT}/{p}")}
    def verified(rows, tag):
        out = {}
        for i, r in rows.items():
            a = r.get("answer")
            out[i] = bool(a) and (J.score(f"{tag}:{i}", i, a if isinstance(a, str) else json.dumps(a)) or 0) >= 1.0
        return out
    cc = verified(idx("evals/solver_results/slr_claude_code/slr.jsonl"), "cc")
    qo = verified(idx("evals/solver_results/slr_qwen3.6_35b_a3b/Sun_Apr_26_131_PM/slr.jsonl"), "qo")
    llm = {"BoK": {i: str(r["solved"]).lower() == "true" for i, r in idx("outputs/slr_bench_best_of_k_stripped.jsonl").items()},
           "DF": {i: str(r["solved"]).lower() == "true" for i, r in idx("outputs/slr_bench_direct_feedback_stripped.jsonl").items()}}
    sd = {i: str(r["seeded_solved"]).lower() == "true" for i, r in idx("outputs/seeded_df_slr_cc.jsonl").items()}
    rows = {"CC Solver": cc, "QO Solver": qo, "BoK": llm["BoK"], "DF": llm["DF"]}
    for L, d in llm.items():
        rows[f"{L} + CC"] = {i: cc[i] or d[i] for i in range(1000)}
        rows[f"{L} + QO"] = {i: qo[i] or d[i] for i in range(1000)}
        rows[f"{L} + CC + QO"] = {i: cc[i] or qo[i] or d[i] for i in range(1000)}
    rows["Seeded DF + CC (lower bound)"] = {i: cc[i] or sd.get(i, False) for i in range(1000)}
    missing = [i for i in range(1000) if not cc[i] and i not in sd]
    rep = {}
    for k, d in rows.items():
        t = {tr: round(100 * sum(d[i] for i in range(1000) if tier[i] == tr) / 250, 1) for tr in ("basic", "easy", "medium", "hard")}
        rep[k] = {"acc": round(sum(d.values()) / 10, 2), "tiers": t}
        print(f"{k:32s} {rep[k]['acc']:6.2f}  {t}")
    print(f"tasks with no seeded run (false CC claims): {len(missing)} -> {missing}")
    rep["_seeded_missing_task_ids"] = missing
    json.dump(rep, open(f"{ROOT}/metrics/slr_hybrids_verified.json", "w"), indent=1)
    J.inq.put(None)
