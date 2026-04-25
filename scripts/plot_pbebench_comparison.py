"""
4-way comparison plots for PBEBench (Lite or Hard):
  DF (gpt-oss-120b), BoK (gpt-oss-120b), CC Solver, OH Qwen Solver.

x-axis: cascade length (number of replace() programs in the GT solution).
Produces 3 figures per benchmark:
  pbebench_{split}_comparison_passrate.png
  pbebench_{split}_comparison_meanreward.png
  pbebench_{split}_comparison_complexity.png   — GT reference from tasks file

Usage:
    # PBEBench-Lite (DF + BoK available):
    python scripts/plot_pbebench_comparison.py --split lite --plot

    # PBEBench-Hard (no DF):
    python scripts/plot_pbebench_comparison.py --split hard --plot

    # Override any path:
    python scripts/plot_pbebench_comparison.py --split lite --plot \\
        --df outputs/lite_tasks_full_og_direct_feedback.jsonl \\
        --bok outputs/lite_tasks_full_og_best_of_k.jsonl
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rewards.pbebench import _parse_programs, _validate_programs, cascade_complexity

# ── default paths ──────────────────────────────────────────────────────────────

DEFAULTS = {
    "lite": {
        "df":           REPO_ROOT / "outputs/lite_tasks_full_og_direct_feedback.jsonl",
        "bok":          REPO_ROOT / "outputs/lite_tasks_full_og_best_of_k.jsonl",
        "cc_solver":    REPO_ROOT / "evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl",
        "qwen_solver":  REPO_ROOT / "evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/lite.jsonl",
        "tasks":        REPO_ROOT / "data/pbebench/lite_tasks_full_og.jsonl",
        "max_programs": 5,
    },
    "hard": {
        "df":           None,  # not available
        "bok":          REPO_ROOT / "outputs/gpt_oss_120b_pbebench_hard_outputs.jsonl",
        "cc_solver":    REPO_ROOT / "evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl",
        "qwen_solver":  REPO_ROOT / "evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard.jsonl",
        "tasks":        REPO_ROOT / "data/pbebench/tasks_full_og.jsonl",
        "max_programs": 20,
    },
}

# Consistent colours + styles across both benchmarks
# (label_prefix, color, is_solver)
SYSTEM_STYLE = {
    "DF (gpt-oss-120b)":  ("#16A34A", False),
    "BoK (gpt-oss-120b)": ("#D97706", False),
    "CC Solver":           ("#2563EB", True),
    "OH Qwen Solver":      ("#DC2626", True),
}

_REPLACE_RE = re.compile(
    r"""replace\(\s*["']([^"']*)["']\s*,\s*["']([^"']*)["']\s*\)""",
    re.IGNORECASE,
)

# ── loaders ────────────────────────────────────────────────────────────────────

def _parse_programs_safe(answer) -> list | None:
    if answer is None:
        return None
    if isinstance(answer, list):
        raw = "\n".join(str(x) for x in answer)
    else:
        raw = str(answer).strip()
        raw = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip("` \n")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                raw = "\n".join(str(x) for x in parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    programs = _REPLACE_RE.findall(raw)
    return programs if programs else None


def _gt_programs_from_str(prog_strs: list[str]) -> list[tuple[str, str]]:
    """Parse GT programs field (list of replace("a","b") strings) → [(pred, transform)]."""
    pairs = []
    for s in prog_strs:
        m = _REPLACE_RE.findall(s)
        pairs.extend(m)
    return pairs


def load_tasks(path: Path) -> dict[int, dict]:
    """Load tasks file → {task_index: {cascade_length, gt_complexity}}."""
    meta = {}
    with open(path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            cl = rec.get("cascade_length")
            progs = _gt_programs_from_str(rec.get("programs", []))
            gt_c = cascade_complexity(progs) if progs else None
            meta[i] = {"cascade_length": cl, "gt_complexity": gt_c}
    return meta


def _score_bok_candidate(candidate: str, inputs: list, outputs: list, max_programs: int) -> float:
    programs, _ = _parse_programs(candidate)
    if programs is None:
        return 0.0
    if _validate_programs(programs, max_programs=max_programs):
        return 0.0
    correct = sum(
        1 for inp, exp in zip(inputs, outputs)
        if _apply(programs, inp) == exp
    )
    return correct / len(inputs) if inputs else 0.0


def _apply(programs, inp: str) -> str:
    s = inp
    for pred, transform in programs:
        s = s.replace(pred, transform)
    return s


def load_bok_raw(path: Path, max_programs: int) -> dict[int, dict]:
    """Load raw {input, outputs} BoK format (hard BoK or any raw sampling file)."""
    records: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            inp = rec["input"]
            idx_str = inp.get("index", "")
            try:
                task_id = int(str(idx_str).split("_")[0])
            except (ValueError, IndexError):
                continue
            candidates = rec.get("outputs", [])
            best_s, best_c = 0.0, None
            for c in candidates:
                s = _score_bok_candidate(c, inp["inputs"], inp["outputs"], max_programs)
                if s > best_s:
                    best_s, best_c = s, c
                if best_s >= 1.0:
                    break
            records[task_id] = {
                "_best_reward": best_s,
                "answer": best_c,
                "cascade_length": inp.get("cascade_length"),
            }
    return records


def load_standard(path: Path) -> dict[int, dict]:
    """Load standard JSONL (LLM direct-feedback or ensemble). Keeps highest-reward per task."""
    records: dict[int, dict] = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("task_index")
            if tid is None:
                tid = i
            br = rec.get("best_reward")
            if br is None:
                br = 1.0 if rec.get("solved") else 0.0
            br = float(br)
            prev = records.get(tid)
            if prev is None or br > float(prev.get("_best_reward", 0)):
                rec["_best_reward"] = br
                records[tid] = rec
    return records


def load_solver(path: Path) -> dict[int, dict]:
    """Load solver JSONL (eval_solver format: score/success/program or best_reward/answer)."""
    records: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("task_index")
            if tid is None:
                continue
            # Prefer continuous score over binary solved/success flags
            if rec.get("best_reward") is not None:
                br = float(rec["best_reward"])
            elif rec.get("score") is not None:
                br = float(rec["score"])
            elif rec.get("solved") is not None:
                br = 1.0 if rec["solved"] else 0.0
            else:
                br = 1.0 if rec.get("success") else 0.0
            rec["_best_reward"] = br
            records[tid] = rec
    return records


# ── GT complexity from tasks meta ─────────────────────────────────────────────

def gt_complexity_by_cl(meta: dict[int, dict]) -> dict[int, float]:
    groups: dict[int, list] = defaultdict(list)
    for m in meta.values():
        cl = m.get("cascade_length")
        gc = m.get("gt_complexity")
        if cl is not None and gc is not None:
            groups[cl].append(gc)
    return {cl: sum(vs) / len(vs) for cl, vs in groups.items()}


# ── per-cascade-length stats ───────────────────────────────────────────────────

def _pred_complexity(rec: dict) -> float | None:
    ans = rec.get("answer") or rec.get("program")
    if ans is None:
        return None
    # Solver may store answer as [[pred, transform], ...] pairs directly
    if isinstance(ans, list) and ans and isinstance(ans[0], (list, tuple)) and len(ans[0]) == 2:
        try:
            return cascade_complexity([(str(p[0]), str(p[1])) for p in ans])
        except Exception:
            pass
    progs = _parse_programs_safe(ans)
    if progs is None:
        return None
    return cascade_complexity(progs)


def stats_by_cl(results: dict[int, dict], meta: dict[int, dict]) -> dict[int, dict]:
    groups: dict[int, list] = defaultdict(list)
    for tid, rec in results.items():
        cl = rec.get("cascade_length") or (meta.get(tid) or {}).get("cascade_length")
        if cl is not None:
            groups[int(cl)].append((tid, rec))

    out = {}
    for cl, items in groups.items():
        n = len(items)
        br_list = [float(rec.get("_best_reward", 0)) for _, rec in items]
        solved = sum(1 for v in br_list if v >= 1.0)
        mean_r = sum(br_list) / n

        pred_cs = []
        for tid, rec in items:
            if float(rec.get("_best_reward", 0)) < 1.0:
                continue
            pc = _pred_complexity(rec)
            if pc is not None:
                pred_cs.append(pc)

        out[cl] = {
            "n": n,
            "pass_rate": solved / n,
            "mean_reward": mean_r,
            "mean_pred_complexity": sum(pred_cs) / len(pred_cs) if pred_cs else None,
        }
    return out


# ── plot helpers ───────────────────────────────────────────────────────────────

def _base_fig():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.5, color="#E5E7EB")
    return fig, ax


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── plots ──────────────────────────────────────────────────────────────────────

def _xlabel(split: str) -> str:
    return "Cascade length (number of replace() programs)"


def _title_suffix(split: str) -> str:
    n_per = {"lite": "252 tasks per level", "hard": "64 tasks per level"}
    return f"PBEBench-{'Lite' if split == 'lite' else 'Hard'}, {n_per[split]}"


def plot_pass_rate(cl_stats: dict[str, dict[int, dict]], all_cls: list[int],
                   split: str, out: Path):
    fig, ax = _base_fig()
    xs = list(range(len(all_cls)))
    for label, by_cl in cl_stats.items():
        color, is_solver = SYSTEM_STYLE[label]
        vals = [by_cl.get(cl, {}).get("pass_rate", float("nan")) * 100 for cl in all_cls]
        ax.fill_between(xs, vals, alpha=0.08, color=color)
        ax.plot(xs, vals, marker="o", markersize=6, linewidth=2.0, color=color,
                linestyle="--" if is_solver else "-", label=label, zorder=3)

    ax.axhline(100, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(50,  color="#9CA3AF", linewidth=0.8, linestyle=":",  zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(cl) for cl in all_cls], fontsize=9)
    ax.set_xlabel(_xlabel(split), fontsize=12)
    ax.set_ylabel("Pass rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_title(f"Pass rate vs cascade length\n({_title_suffix(split)})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    _save(fig, out)


def plot_mean_reward(cl_stats: dict[str, dict[int, dict]], all_cls: list[int],
                     split: str, out: Path):
    fig, ax = _base_fig()
    xs = list(range(len(all_cls)))
    for label, by_cl in cl_stats.items():
        color, is_solver = SYSTEM_STYLE[label]
        vals = [by_cl.get(cl, {}).get("mean_reward", float("nan")) for cl in all_cls]
        ax.fill_between(xs, vals, alpha=0.08, color=color)
        ax.plot(xs, vals, marker="o", markersize=6, linewidth=2.0, color=color,
                linestyle="--" if is_solver else "-", label=label, zorder=3)
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}", xy=(x, v), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7, color="#374151")

    ax.axhline(1.0, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(0.5, color="#9CA3AF", linewidth=0.8, linestyle=":",  zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(cl) for cl in all_cls], fontsize=9)
    ax.set_xlabel(_xlabel(split), fontsize=12)
    ax.set_ylabel("Mean reward", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title(f"Mean reward vs cascade length\n({_title_suffix(split)})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    _save(fig, out)


def plot_complexity(cl_stats: dict[str, dict[int, dict]], all_cls: list[int],
                    gt_by_cl: dict[int, float], split: str, out: Path):
    fig, ax = _base_fig()
    xs = list(range(len(all_cls)))

    gt_vals = [gt_by_cl.get(cl, float("nan")) for cl in all_cls]
    if any(not np.isnan(v) for v in gt_vals):
        ax.plot(xs, gt_vals, marker="s", markersize=5, linewidth=1.5,
                color="#9CA3AF", linestyle=":", label="GT", zorder=2)

    for label, by_cl in cl_stats.items():
        color, is_solver = SYSTEM_STYLE[label]
        vals = [by_cl.get(cl, {}).get("mean_pred_complexity") for cl in all_cls]
        vals = [v if v is not None else float("nan") for v in vals]
        ax.fill_between(xs, vals, alpha=0.08, color=color)
        ax.plot(xs, vals, marker="o", markersize=6, linewidth=2.0, color=color,
                linestyle="--" if is_solver else "-", label=label, zorder=3)
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.1f}", xy=(x, v), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7, color="#374151")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(cl) for cl in all_cls], fontsize=9)
    ax.set_xlabel(_xlabel(split), fontsize=12)
    ax.set_ylabel("Mean cascade complexity (correct solutions)", fontsize=12)
    ax.set_title(f"Cascade complexity vs cascade length\n({_title_suffix(split)}, correct solutions only)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, out)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["lite", "hard"], required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--figures-dir", default=str(REPO_ROOT / "figures"))
    parser.add_argument("--df",          default=None)
    parser.add_argument("--bok",         default=None)
    parser.add_argument("--cc-solver",   default=None)
    parser.add_argument("--qwen-solver", default=None)
    parser.add_argument("--tasks",       default=None)
    parser.add_argument("--metrics-json", default=None)
    args = parser.parse_args()

    cfg = DEFAULTS[args.split]
    df_path   = Path(args.df)          if args.df          else cfg["df"]
    bok_path  = Path(args.bok)         if args.bok         else cfg["bok"]
    cc_path   = Path(args.cc_solver)   if args.cc_solver   else cfg["cc_solver"]
    qw_path   = Path(args.qwen_solver) if args.qwen_solver else cfg["qwen_solver"]
    tasks_path = Path(args.tasks)      if args.tasks       else cfg["tasks"]
    max_prog  = cfg["max_programs"]
    fdir      = Path(args.figures_dir)

    if args.plot:
        fdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tasks: {tasks_path}")
    meta = load_tasks(tasks_path)
    gt_by_cl = gt_complexity_by_cl(meta)

    # Determine if BoK is raw format (has "input"/"outputs" keys) vs standard JSONL
    def _is_raw_bok(p: Path) -> bool:
        with open(p) as f:
            first = json.loads(f.readline())
        return "input" in first and "outputs" in first

    systems: dict[str, dict[int, dict]] = {}

    if df_path and df_path.exists():
        print(f"Loading DF: {df_path}")
        systems["DF (gpt-oss-120b)"] = load_standard(df_path)
        print(f"  {len(systems['DF (gpt-oss-120b)'])} tasks")
    elif df_path:
        print(f"  Warning: DF not found at {df_path}, skipping.")

    if bok_path and bok_path.exists():
        print(f"Loading BoK: {bok_path}")
        if _is_raw_bok(bok_path):
            systems["BoK (gpt-oss-120b)"] = load_bok_raw(bok_path, max_prog)
        else:
            systems["BoK (gpt-oss-120b)"] = load_standard(bok_path)
        print(f"  {len(systems['BoK (gpt-oss-120b)'])} tasks")
    elif bok_path:
        print(f"  Warning: BoK not found at {bok_path}, skipping.")

    if cc_path and cc_path.exists():
        print(f"Loading CC Solver: {cc_path}")
        systems["CC Solver"] = load_solver(cc_path)
        print(f"  {len(systems['CC Solver'])} tasks")
    elif cc_path:
        print(f"  Warning: CC Solver not found at {cc_path}, skipping.")

    if qw_path and qw_path.exists():
        print(f"Loading OH Qwen Solver: {qw_path}")
        systems["OH Qwen Solver"] = load_solver(qw_path)
        print(f"  {len(systems['OH Qwen Solver'])} tasks")
    elif qw_path:
        print(f"  Warning: OH Qwen Solver not found at {qw_path}, skipping.")

    if not systems:
        print("No data loaded — nothing to do.")
        return

    print("Computing per-cascade-length stats...")
    cl_stats = {name: stats_by_cl(res, meta) for name, res in systems.items()}

    # All cascade lengths covered by any system
    all_cls = sorted(set(cl for by_cl in cl_stats.values() for cl in by_cl))

    # Print summary table
    print(f"\n{'System':<25} {'CL':>4} {'N':>5} {'Pass%':>7} {'MeanRew':>9} {'MeanComplexity':>15}")
    for name, by_cl in cl_stats.items():
        for cl in all_cls:
            s = by_cl.get(cl)
            if not s:
                continue
            mc = f"{s['mean_pred_complexity']:.2f}" if s["mean_pred_complexity"] is not None else "—"
            print(f"  {name:<23} {cl:>4} {s['n']:>5} {s['pass_rate']*100:>6.1f}% "
                  f"{s['mean_reward']:>9.4f} {mc:>15}")
        print()

    if args.metrics_json:
        out_metrics = {}
        for name, by_cl in cl_stats.items():
            out_metrics[name] = {str(cl): v for cl, v in by_cl.items()}
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w") as f:
            json.dump(out_metrics, f, indent=2)
        print(f"Metrics written to {args.metrics_json}")

    if args.plot:
        split = args.split
        print("Generating figures...")
        plot_pass_rate(cl_stats, all_cls, split,
                       fdir / f"pbebench_{split}_comparison_passrate.png")
        plot_mean_reward(cl_stats, all_cls, split,
                         fdir / f"pbebench_{split}_comparison_meanreward.png")
        plot_complexity(cl_stats, all_cls, gt_by_cl, split,
                        fdir / f"pbebench_{split}_comparison_complexity.png")


if __name__ == "__main__":
    main()
