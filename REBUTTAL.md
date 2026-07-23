# ReaComp — Rebuttal / Author Response

We thank all three reviewers for their careful and constructive reviews. Below we
respond to **every point each reviewer raised, in the order they raised it**
(Weaknesses, Questions, Clarity, and Limitations notes), quoting each concern so
it is easy to verify that we have addressed it. Items awaiting rebuttal-period
experiments are marked **[TBD]** and will be filled in with concrete numbers
before discussion closes.

One request is shared by all three reviewers — a comparison against other
neuro-symbolic / library-learning / solver-induction methods. We are adding a
**TroVE** baseline (Wang et al., 2024b, already cited in §6) run in the *same*
Qwen3.6-35B-A3B + OpenHands + verifier harness as our DirectSolve baseline, so
any gap is attributable to the induction mechanism rather than to model or
infrastructure differences. We refer to this as **[R-TROVE]** below and report
**[TBD: numbers]** for PBEBench-Lite and SLR-Bench.

---

# Reviewer ggna (Rating: 4 — Borderline Accept)

### Weakness 1 — "The quality of the reasoning traces impacts the quality of the induced solvers and their ability to solve problems at inference time without requiring LLMs."

We agree, and we already have direct evidence quantifying this dependence:
- Removing chain-of-thought from the demonstrations (no-CoT ablation, App. F.2)
  drops PBEBench-Hard accuracy from **74.7% → 24.8%**, whereas *reducing the
  number* of CoT examples (100 → 48) has a much smaller effect. Solver quality is
  therefore governed by the *reasoning content* of traces, not their volume.
- We will add a dedicated Limitations paragraph on trace quality (see the reviewer's
  Limitations note below).

### Weakness 2 — "The induced solvers come from a dataset that is likely in the domain of the LLM's knowledge … the method still has test-time-scaling issues to novel tasks or variations that the LLM (and now solvers) has not seen."

This is a fair concern about generalization, which we address empirically:
- The **forward-reconstruction case study** (§4, App. F.4) is an *out-of-domain*
  test: real historical-linguistics data with an **unseen IPA alphabet**,
  variable/unknown cascade lengths, and **no ground-truth programs**. Induced
  solvers transfer **zero-shot** at ~70% individually and **80.1%** ensembled —
  genuine distribution shift, not resampling the training distribution.
- Because the solver is symbolic and standalone, it makes **no LLM calls at test
  time**, so on novel inputs it cannot silently regress to LLM priors — it either
  generalizes algorithmically or fails transparently (the 4% "unique-op" solver
  under shift, App. F.4, illustrates the honest failure mode).
- We concede solvers are bound to distributions sharing the DSL's structural
  assumptions and state this in Limitations (App. B).

### Weakness 3 — "How does the method handle reasoning traces that have failed resolutions/solutions (and similarly reasoning traces that have noise)?"

By design, the trace set **intentionally includes failed attempts**: it is
balanced across the difficulty × outcome quadrants (easy/hard × success/failure),
seed-42 sampled (§2.2, F.2). The building prompt explicitly directs the agent to
"take inspiration … especially in cases where the LLM struggles" (App. D.1), and
the trajectory analysis (App. F.2) shows the agent extracting *failure modes* from
unsuccessful traces (e.g., the "safety-first" strategy that learns *not* to modify
already-correct examples). The no-CoT result is the decisive evidence: when only
final programs (success signal) are given without reasoning, performance collapses
— i.e., **outcomes alone are insufficient supervision; the reasoning in both
successful and failed traces carries the signal.** We will make this explicit.

### Weakness 4 — "Table 1, BoK with symbolic solvers shows similar performance as the BoK baseline (93.8) while the token cost is significantly reduced, the monetary costs are higher (why would this happen?). Similarly for the DF-32 baseline."

Great catch — this is a presentation artifact we will fix. The **token** column for
hybrids counts **LLM fallback tokens only** (tasks solved by the zero-cost solver
contribute 0 tokens). The **cost** column *additionally* folds in the **one-time
solver construction cost** ($2.00 CC, $0.85 QO). On near-saturated Lite, BoK is
already at ceiling, so the hybrid saves inference tokens but the added
construction line item can make the *total* dollar figure tie or slightly exceed
BoK — even though per-task inference is cheaper. This is exactly why we frame Lite
as an **efficiency** regime (§5), and why the large cost wins appear on **Hard**
(71.6M vs 332.1M tokens, −78%). Fix: we will (a) split "inference cost" and
"amortized construction cost" into separate columns, and (b) note that
construction cost is a fixed one-time charge that vanishes as task count grows.

### Clarity — "DSL is not defined."

Both DSLs are in fact defined in the main text, in **§3** (Experimental Setup):
"programs are `replace(A,B)` sequences with `1 ≤ |A| ≤ 3`, `0 ≤ |B| ≤ 3`,
evaluated by exact execution; for SLR-Bench, solutions are Prolog rules
`eastbound(T) :- Body.`" The abstract DSL `P` is also introduced in the §2 problem
formulation ("infer a program `p ∈ P`"). We suspect the issue is **salience and
placement**: the abstract notation `P` appears in §2 roughly a page before the
concrete DSL is spelled out in §3, so a reader meets the symbol before its
grounding. To fix this we will (i) attach the concrete `replace(A,B)` / Prolog-rule
definitions to the first mention of `P` in §2, and (ii) add a one-line boxed "DSL"
definition (with the `|A|`, `|B|`, predicate-set constraints) at the top of §3 so
it cannot be missed. The full specifications remain in App. D.1 / F.1 / F.6.

### Clarity — "It would be easier to interpret the results if the best results were highlighted consistently across the whole table, not just the paper's results."

Agreed. We will apply one consistent convention — **best value per column across
all rows (baselines included)** — in Tables 1–3.

### Question 1 — "How sensitive is ReaComp to the quality and diversity of the initial reasoning traces?"

See Weaknesses 1 and 3. In short: highly sensitive to *reasoning content*
(no-CoT: 74.7% → 24.8% on Hard), much less sensitive to *count* (48 vs 100 CoT
examples comparable); diversity is enforced by the balanced quadrant sampling; and
run-to-run *algorithmic* variance (53.4–79.2% Lite on identical traces) dominates
dataset size — which ensembling recovers (91.3% Lite / 84.7% Hard).

### Question 2 — "Are there other recent neuro-symbolic or solver-induction methods that could serve as stronger baselines? … Could the authors clarify why these methods were not included as baselines, and how ReaComp differs empirically from them?"

See **[R-TROVE]**. We are adding TroVE under the matched harness, and will add
text bridging §6/App. G to the evaluated baselines. The methodological distinction:
DreamCoder/Stitch/LILO/ReGAL/TroVE/ToolLibGen induce *reusable components invoked
by an LLM or search at inference time*, whereas ReaComp compiles reasoning into a
**standalone solver that runs with zero LLM calls**. The Sesterhenn et al. (2025)
compute-matched re-evaluation we already cite reinforces this: library-learning
gains often shrink under compute control — a critique ReaComp sidesteps by moving
cost entirely offline. Empirically, our existing **DirectSolve** baseline (same
Qwen3.6 agent, per task: 87.2% Lite / 24.9% Hard / 58.4% SLR) already shows that
per-task agentic effort trails the compiled solver on hard instances; TroVE will
add the cross-task library-reuse comparison. **[TBD: numbers]**

### Limitations note — "It seems to be missing a discussion on the quality of reasoning traces."

We will add a Limitations paragraph specifically on reasoning-trace quality,
anchored to the no-CoT ablation and the success/failure-balanced trace protocol.

---

# Reviewer ZhtM (Rating: 5 — Accept)

### Weakness — "The induced solvers are specific to the domain of the training tasks. If the test tasks are too far from the train distribution, I suspect the method cannot adapt."

Agreed as a scoped limitation; see the forward-reconstruction evidence under
ggna-W2 (zero-shot to unseen IPA alphabet / real data, ~70% single, 80.1%
ensembled). Solvers generalize across distributions that share the DSL's
structural assumptions, and we state the boundary explicitly in App. B and Q5
below.

### Question 1 — "With which model are the reasoning traces created, based on which the coding agents operate? Is it always the same?"

**`gpt-oss-120b`**, for both PBEBench and SLR-Bench, and it is the **same** across
all induction runs and ablations (§2.2, §3). Note the distinction we will make
clearer: the *trace-generating* model (`gpt-oss-120b`) is separate from the
*inducing coding agent* (Claude Code `claude-sonnet-4-6`, or Qwen3.6-35B-A3B via
OpenHands).

### Question 2 — "Can SOLVER.py have any form? Do you constrain it somehow … how does the example template look, how much bias do you put in there?"

The solver is **interface-constrained, not algorithm-constrained**. The building
prompt (App. D.1) fixes only: the function signature
(`solve_pbe(examples)` / `solve_slr(examples)` returning `{success, program, ...}`),
read-only verifier access, the DSL spec, stdlib-only, and *soft* preferences
("prefer simple, compositional rules"). It does **not** prescribe a search
algorithm. Evidence that little algorithmic bias is injected: three runs on
identical inputs invent qualitatively different algorithms — greedy + residual
fixing, safety-first greedy + 2-step lookahead, unique-op permutations, beam
search (App. F.2, Table 13). We will point to the full template in App. D.1 and
state explicitly what is/isn't constrained.

### Question 3 — "What if the search space inside the solver is very large? Do you restrict execution time? Could a solver find the correct program but be cut off before?"

Solvers do **not** enumerate the space (quantified as up to ~10^206 for Hard,
App. F.1/F.3); they extract candidate predicates from input-output diffs and run
bounded beam/greedy search over hundreds-to-thousands of candidates per task. The
eval harness also imposes a per-task timeout (`--task-timeout`, e.g. 60s PBE /
3600s SLR). Evidence that *depth*, not the time cut-off, is the binding
constraint: on real-FR, raising max cascade length 20 → 100 gives only ~1pp
(App. F.4), and per-length breakdowns (Tables 12, 16) show graceful degradation
with failures being **near-misses** (mean reward > 0.92 even at CL 20). We will
report exact timeout values and the fraction of tasks that hit them. **[TBD:
timeout-hit counts]**

### Question 4 — "Could you provide one or more qualitative examples of a reasoning trace, the discovered DSL from it (or parts of it), and its usage for a final program?"

Yes — we will add a worked end-to-end figure: (i) an excerpt of a `gpt-oss-120b`
trace reasoning about edit regions, (ii) the induced solver mechanism it inspired
(e.g., extract edit regions via `difflib.SequenceMatcher`; restrict candidates to
observed diffs; safety-first beam), and (iii) a concrete task the resulting solver
solves. App. F.4 already lists recovered real rules (e.g. `replace('#b','#f')` =
/b/→/f/ lenition; `replace('ʔ','')` = glottal-stop deletion) we can draw from.

### Question 5 — "Do you agree that the solver is bound to the domain of the training tasks and can only generalize within their scope, or am I missing something?"

Largely yes, with an important qualifier: generalization holds across
*distributions that share the DSL's structural assumptions*, not arbitrary new
task formats. The forward-reconstruction study demonstrates real within-family
generalization under distribution shift (unseen alphabet, sparse examples, no gold
programs). We state this boundary in App. B and will make it crisper in the main
text.

### Question 6 — "Could it be an option to have a test task, create one or more reasoning traces for it (where the LLM fails but has partially correct ideas), then aggregate these to get a solver? … weigh against the cost of the coding agent."

We find this a genuinely interesting direction and thank the reviewer. It reframes
ReaComp as a *per-instance-family* compiler: generate a few (possibly failed)
traces on the target family, then induce a small solver from them — trading
coding-agent construction cost against per-task LLM cost, most attractive when many
structurally-similar tasks share a family. We will add it to Future Work with the
cost trade-off noted. **[TBD: small proof-of-concept from held-out-task traces if
time permits; otherwise discussed as Future Work]**

### Suggestion — "The notation of candidates in lines 83 and 85 is a bit vague; it could be more formally clean."

We will tighten it: define `C = {p_k}_{k=1}^K` for BoK and `C = {p^(t)}_{t=1}^T`
for DF, and use consistent `C_S` / `C_L` for solver vs. LLM candidate sets in
Alg. 2.

### Suggestion — "Mixing the ablations into the main table was confusing/overloading … evaluate your method alone first (one solver), then introduce the bag of solvers (All symbolic) separately."

Agreed. We will restructure: the main table reports primary systems in escalating
order (single solver → CC+QO → All Symbolic), and the 6-run ablation grid moves to
its own labeled table/section.

### Suggestion — "Table 2 and 3 have inconsistent bold markings (Table 2 bolds costs that are not the lowest; Table 3 has no bold numbers)."

We will fix bolding across all result tables to a single "best per column"
convention.

### Suggestion — "Related work could mention Vision-Language Programs [1] and ActivationReasoning [2]."

Thank you. We will add both: Vision-Language Programs (Wüst et al., CVPR 2026),
combining VLM functions with task-specific induced primitives; and
ActivationReasoning (Helff et al., ICLR 2026), logical reasoning over SAE-derived
activation concepts (we note it shares an author with the SLR-Bench benchmark we
build on).

---

# Reviewer orXJ (Rating: 2 — Reject, Confidence: 5)

### Concern 1 — "The symbolic program synthesizer is generated using only a simple one-paragraph prompt … Therefore it is difficult to verify whether the resulting symbolic solver is actually strong or well-designed."

We respectfully clarify a misreading. The one sentence quoted is the *task line* of
a substantially longer, structured specification (**Appendix D.1**) that also fixes
the solver interface, grants **read-only verifier access** so the agent can test
and debug candidate solvers, specifies the DSL, restricts to the standard library,
and states behavioral requirements (prefer low-complexity compositional rules,
return top-K on failure). The solver is **not** produced from one paragraph in one
shot: it is the output of a **multi-step agentic construction loop** (Alg. 1) with
code edits, execution against the trace set, and reward-guided refinement over
**49–102 agent turns** per run (App. C, Table 14). Moreover, solver *strength* is
not asserted but **verifier-established**: every "solved" is `reward = 1.0` under
exact deterministic execution (§3), with per-cascade-length, per-tier, and
per-BFCC breakdowns (Tables 12, 16, 17, 27) and statistical-significance testing
(App. F.5). We will make the App. D.1 pointer prominent in the main text.

### Concern 2 — "A comparison with other symbolic solvers is necessary."

Agreed; see **[R-TROVE]**. We are adding a TroVE library-induction baseline in the
identical Qwen3.6 + OpenHands + verifier harness on PBEBench-Lite and SLR-Bench.
Together with the existing DirectSolve coding-agent baseline (87.2% Lite / 24.9%
Hard / 58.4% SLR), this brackets ReaComp against both per-task agentic solving and
cross-task library learning under matched conditions. **[TBD: numbers]**

### Concern 3 — "For the fallback cases, the authors appear to solve them directly using an LLM. It would be worth considering how to better leverage the outputs produced by the symbolic solver when handling these fallback cases."

A fair, constructive point.
- The solver *already* returns structured partial hypotheses (top-K scored
  programs) even on failure — the interface is designed for downstream refinement
  (App. D.1). In the current *effi* hybrid we select best-by-reward across solver
  and LLM outputs, but we do **not** yet *seed* the LLM fallback with the solver's
  near-miss.
- The reviewer's suggestion — warm-starting LLM search from the solver's top-K
  near-misses (as DF's initial trajectory / few-shot context) — is promising,
  especially since solver failures are overwhelmingly **near-misses** (mean reward
  0.987 on Hard where accuracy is low; App. F.3). **[TBD: solver-seeded-fallback
  ablation on a subset if time permits; otherwise a concrete Future Work item
  motivated by the near-miss statistics]**

### Question — "Given prior attempts to formulate such problems as MDPs and solve them autoregressively [1], or through latent reasoning [2], it would be worth considering what methodological extensions or more advanced formulations could be developed."

We will engage both in Related Work / Discussion. ARCLE [1] frames inductive
reasoning as an RL environment — an *online, per-task policy-learning* view;
ReaComp is complementary, moving learned structure **offline** into a reusable
symbolic artifact with zero per-task inference. Latent-reasoning [2] keeps
computation in activations; ReaComp externalizes it into inspectable, verifiable
code. Solver-seeded fallback (Concern 3) is one concrete methodological extension
toward combining ReaComp with these; we will add a positioning paragraph.

### Question — "The paper uses PBEBench and SLR-Bench … other inductive reasoning benchmarks such as Mini-ARC, ARC-AGI-1/2, MiniSCAN, List Functions, and ACRE can also be viewed as Programming-by-Example … it may be worthwhile to expand the experiments using the benchmarks discussed in [3]."

We agree these are natural next targets and thank the reviewer for the pointer to
the hypothesis-refinement study [3]. Our claims are deliberately scoped to two
*executable, verifier-backed, procedurally-generated* domains plus one *real-world*
transfer study (historical linguistics), spanning synthetic and natural
distributions. Extending to ARC-family / List-Functions requires per-domain DSLs
and verifiers and is beyond a responsible rebuttal-window addition; we will add it
to Future Work and scope any over-broad phrasing to verifier-backed PBE/ILP
settings. We note the forward-reconstruction study already exhibits the ARC-like
few-shot inductive structure the reviewer highlights, on real data.

### On the overall assessment (Quality: 1 → we hope to move this)

The core contribution — compiling LLM reasoning traces into standalone symbolic
solvers that (i) run at zero per-task LLM cost, (ii) **outperform** LLM test-time
scaling on hard, long-horizon instances (+16.3pp over BoK on PBEBench-Hard;
matching frontier o3/GPT-5 on the SLR Hard tier at zero inference cost), and (iii)
Pareto-dominate per-task agentic effort while cutting tokens up to 78% — is, to our
knowledge, novel, and is supported by verifier-exact evaluation with significance
testing (App. F.5). We hope the clarification of the construction procedure
(Concern 1) and the new matched baseline (Concern 2) substantively address the
reviewer's concerns and merit reconsideration.

---

## Summary of paper changes

1. **New TroVE baseline** (matched Qwen3.6 + OpenHands harness), Lite + SLR. **[TBD]**
2. **Improve DSL salience**: move the concrete `replace(A,B)` / Prolog-rule
   definition (already in §3) next to the first mention of `P` in §2 + add a boxed
   §3 definition. *(ggna)*
3. **Split cost columns**: inference vs. amortized one-time construction; resolve
   the Table 1 token-vs-cost question. *(ggna-W4)*
4. **Consistent bolding** (best per column) across Tables 1–3. *(ggna, ZhtM)*
5. **Disentangle ablations** from the main table into a dedicated section. *(ZhtM)*
6. **Tighten notation** for `C`, `C_S`, `C_L` (lines 83–85, Alg. 2). *(ZhtM)*
7. **New Limitations paragraph** on reasoning-trace quality / failed-noisy traces.
   *(ggna)*
8. **Worked qualitative example**: trace → induced solver mechanism → solved task.
   *(ZhtM-Q4)*
9. **Clarify**: trace model (`gpt-oss-120b`) vs. inducing agent; solver is
   interface- not algorithm-constrained; execution/time bounds. *(ZhtM, orXJ)*
10. **New Related Work**: Vision-Language Programs, ActivationReasoning, ARCLE,
    latent-reasoning; position ReaComp vs. MDP/latent formulations. *(ZhtM, orXJ)*
11. **Future Work**: solver-seeded LLM fallback; per-instance-family induction from
    test-task traces; ARC-family / List-Functions benchmarks. *(ZhtM-Q6, orXJ)*
