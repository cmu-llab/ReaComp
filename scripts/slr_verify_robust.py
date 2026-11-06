"""Crash-tolerant re-verification of SLR-Bench answers with the official verifier.

Some stored answers make the SLR judge abort the whole Python process (C++ std::length_error /
bus error). This runs the judge in a long-lived worker process; if the worker dies on an item,
that item is recorded as CRASH (not counted as solved) and the worker is restarted.

Usage: python scripts/slr_verify_robust.py <answers.jsonl> [more.jsonl ...]
Writes metrics/reverified_slr_robust.json with self-reported / verified accuracy and crash ids.
"""
import json, multiprocessing as mp, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]


def worker(inq, outq):
    sys.path.insert(0, ROOT)
    from rewards.slr_bench import reward
    tasks = jl(f"{ROOT}/data/slr_bench/v1_All_full.jsonl")
    outq.put(("ready", None, None))
    while True:
        item = inq.get()
        if item is None:
            return
        key, i, ans = item
        try:
            v = reward(ans, True, tasks[i])["value"]
        except Exception as e:  # ordinary Python errors are a non-solve, not a crash
            v = -1.0 if "length" in str(e) else 0.0
        outq.put(("res", key, v))


class Judge:
    def __init__(self):
        self.start()

    def start(self):
        self.inq, self.outq = mp.Queue(), mp.Queue()
        self.p = mp.Process(target=worker, args=(self.inq, self.outq), daemon=True)
        self.p.start()
        assert self.outq.get(timeout=3600)[0] == "ready"

    def score(self, key, i, ans, timeout=120):
        self.inq.put((key, i, ans))
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                kind, k, v = self.outq.get(timeout=1)
                if k == key:
                    return v
            except Exception:
                if not self.p.is_alive():
                    self.start(); return None  # crashed on this item
        self.p.kill(); self.start(); return None  # hung on this item


def main(files):
    out_path = f"{ROOT}/metrics/reverified_slr_robust.json"
    rep = json.load(open(out_path)) if os.path.exists(out_path) else {}
    judge = Judge()
    for f in files:
        rel = os.path.relpath(os.path.abspath(f), ROOT)
        s = v = 0; crashes = []
        for r in jl(f):
            i = int(r["task_index"]); s += float(r.get("best_reward") or 0) >= 1.0
            a = r.get("answer")
            if not a:
                continue
            a = a if isinstance(a, str) else json.dumps(a)
            res = judge.score(f"{rel}:{i}", i, a)
            if res is None:
                crashes.append(i)
            elif res >= 1.0:
                v += 1
        rep[rel] = {"self_reported_acc": round(s / 10, 2), "verified_acc": round(v / 10, 2), "crash_task_ids": crashes}
        print(f"{rel:70s} self {s/10:6.2f} verified {v/10:6.2f} crashes {len(crashes)}", flush=True)
        json.dump(rep, open(out_path, "w"), indent=1)
    judge.inq.put(None)


if __name__ == "__main__":
    main(sys.argv[1:])
