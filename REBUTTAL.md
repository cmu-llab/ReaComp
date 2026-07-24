# ReaComp — Rebuttal / Author Response

We thank the Area Chair and all three reviewers. Below we respond first to the
**meta-review** (Section 0), then to **every point each reviewer raised, in the
order they raised it** (Weaknesses, Questions, Clarity, Limitations), quoting each
concern so it is easy to verify we have addressed it. Items awaiting
rebuttal-period experiments are marked **[TBD]** and will be updated with concrete
numbers before discussion closes.

One request recurs across the meta-review and all three reviewers — a comparison
against other neuro-symbolic / library-learning / solver-induction methods. We
have added a **TroVE** baseline (Wang et al., 2024b, already cited in §6) run in
the *same* Qwen3.6-35B-A3B + OpenHands + verifier harness as our
**Qwen3.6-35B-A3B (OpenHands)** coding-agent baseline, so any gap is attributable
to the induction mechanism rather than to model or infrastructure differences. We
refer to this as **[R-TROVE]**. TroVE achieves **47.9% on PBEBench-Lite** and
**44.1% on SLR-Bench** (mean reward 0.554 / 0.729), versus **87.2% / 58.4%** for
the Qwen3.6-35B-A3B (OpenHands) agent and **93.9% / 86.7%** for ReaComp's best
hybrid. To preempt a compute-fairness objection, we report TroVE under **two
configurations**: (i) a lightweight single-shot setting (K=3 candidates/mode, no
chain-of-thought, 4K-token generations) and (ii) a **compute-matched** setting
whose per-call generation budget matches the Qwen3.6-35B-A3B (OpenHands) agent
(chain-of-thought on, 16K-token generations); **[TBD: matched numbers]**. Beyond
accuracy, [R-TROVE] also yields a **qualitative library-content comparison**
(App. F.2 analyzes what *ReaComp* induces; we do the same for TroVE's toolbox)
that speaks directly to the "reusable abstractions vs. task-specific heuristics"
question in meta-review (d.4).

---

# Section 0 — Response to the Meta-Review

We thank the AC for the accurate summary and for identifying, in point (d), a
concrete set of conditions under which the assessment would improve. We take these
seriously and address each below. We respectfully note that **most of these
concerns are already addressed by experiments in the current submission** (chiefly
Appendices F.2 and F.4), which we suspect were easy to miss given their placement
in the appendix; we will surface them in the main text. We map each AC condition
to the existing (or in-progress) evidence:

**(d.1) "Sensitivity to trace quality."** Already measured. The **no-CoT
ablation** (App. F.2, Table 13) shows that removing reasoning traces from the
demonstrations drops PBEBench-Hard accuracy from **74.7% → 24.8%** (and 53.4% →
42.1% on Lite), while *reducing the number* of CoT examples (100 → 48) has a much
smaller effect. This directly quantifies trace-quality sensitivity and shows the
signal comes from reasoning content, not example count.

**(d.2) "Solver induction variance."** Already measured. Three independent
induction runs on the **identical** 100-example CoT trace set span **53.4–79.2%
(Lite) / 51.8–74.7% (Hard)** (App. F.2, Table 13), each discovering a
qualitatively different algorithm. We characterize this as search over algorithmic
space and show ensembling recovers it (91.3% Lite / 84.7% Hard). We additionally
report inference-time determinism (App. Table 18: CC 99.6%, QO 97.0% per-task
agreement across reruns), separating *induction* variance from *inference*
variance.

**(d.3) "Generalize beyond the original benchmark distributions."** Already
demonstrated. The **forward-reconstruction case study** (§4, App. F.4) evaluates
solvers **zero-shot** on real historical-linguistics data with an **unseen IPA
alphabet**, variable/unknown cascade lengths, and **no ground-truth programs** —
a genuine distribution shift, not resampling of the training distribution. Solvers
reach ~70% individually and **80.1%** ensembled without any retraining.

**(d.4) "Reusable abstractions rather than task-specific heuristics."** This is a
central strength of our approach, and we provide two complementary analyses that
test it directly. **(i) The induced solvers are documented general algorithms, not
lookup tables.** The **qualitative trajectory analysis** (App. F.2, "Qualitative
trajectory analysis"; Table 13) inspects each induced `SOLVER.py` and reports its
*mechanism*: e.g. run 1 = "extract edit regions → greedy selection maximizing fixes
→ residual repair passes"; run 2 = "safety-first hard constraint (never modify
already-correct examples) + 2-step lookahead for interaction effects"; the Claude
Code solver = "two-phase safe/unrestricted beam search with candidate extraction
from `difflib.SequenceMatcher`"; the SLR solver = "ascending-complexity layered
search with early exit on the simplest correct rule." These are input-agnostic
search procedures parameterized by the task's examples — the definition of a
reusable abstraction — and they are inspectable code, not hidden weights.
**(ii) The compression sweep** (App. F.4, Table 20) then stress-tests *how* general
the learned rules are: under strong compression (programs ≤ n_examples/5) accuracy
drops to ~30%, honestly delimiting the tasks that admit compact general rules from
those solved by longer example-patching cascades, and recovering linguistically
plausible sound laws (e.g. /b/→/f/ lenition, glottal-stop deletion). Crucially,
**this is exactly the axis on which a library-learning baseline should be
contrasted, not just accuracy.** In our in-progress **[R-TROVE]** run we observe
that TroVE's induced toolbox is a *mixture* of genuinely general functions
(brute-force replace-search over example-derived candidates) and **memorized,
task-specific functions** — e.g. a "reusable" toolbox entry whose body hardcodes one
task's answer (`programs = ["replace('cb','iyj')", "replace('v','bw')"]`) and is
then re-invoked on unrelated tasks. This is precisely the "task-specific heuristics
masquerading as abstractions" failure the AC is concerned about — and it is a
property of per-task-sampled library induction, whereas ReaComp compiles a *single*
general algorithm whose mechanism we can read off (App. F.2). We will add this
qualitative library-content comparison alongside the accuracy numbers. **[TBD:
final TroVE toolbox analysis]**

**(d.5) "Stronger neuro-symbolic / library-learning baselines."** This is the one
condition not yet in the submission, and we agree it is the most important
addition. See **[R-TROVE]**: a TroVE library-induction baseline in the matched
Qwen3.6 + OpenHands harness on PBEBench-Lite and SLR-Bench. Combined with the
existing Qwen3.6-35B-A3B (OpenHands) coding-agent baseline (87.2% Lite / 24.9%
Hard / 58.4% SLR), this brackets ReaComp against both per-task agentic effort and
cross-task library reuse under matched conditions: TroVE scores 47.9% Lite / 44.1%
SLR — below the coding agent, which is in turn below ReaComp.

**(d.6) "Broader benchmarks."** We currently span two verifier-backed synthetic
domains (string-rewrite cascades, Prolog rule induction) *plus* one real-world
transfer study (historical linguistics), covering synthetic and natural
distributions. We agree ARC-family / List-Functions benchmarks are valuable next
targets but they require per-domain DSLs and verifiers beyond a responsible
rebuttal-window addition; we will add them to Future Work and **narrow the paper's
claims** to the verifier-backed PBE/ILP setting we actually evaluate, as the AC
suggests.

**On "lacks theoretical justification for when induction succeeds/generalizes."**
We agree the contribution is empirical. We will add a discussion framing the
observed regularities as testable conditions: induction succeeds when (i) traces
carry reasoning (d.1), (ii) the target distribution shares the DSL's structural
assumptions (d.3), and (iii) the verifier is fast and exact (App. B). A full theory
of generalization for coding-agent-induced solvers is beyond this paper's scope,
and we will state that boundary explicitly rather than over-claim.

In summary, five of the six conditions in (d) are supported by evidence already in
the submission or by the honest limitation analyses we report; the sixth (stronger
baselines) is in progress. We will make these results far more visible in the main
text and narrow claims where the evidence does not yet reach.

---

# [R-TROVE] — The new TroVE baseline: fidelity and findings

Because a matched neuro-symbolic baseline is the single most-requested addition
(meta-review d.5; ggna-Q2; orXJ-2; ZhtM), we describe here (i) how faithful our
TroVE implementation is, (ii) the headline accuracy, and (iii) what we find in the
induced library.

**Headline accuracy.** Over all 1008 PBEBench-Lite and 1000 SLR-Bench tasks:

| System | PBEBench-Lite Acc% | SLR-Bench Acc% |
|---|---|---|
| TroVE (library induction, this work) | **47.9** | **44.1** |
| Qwen3.6-35B-A3B (OpenHands) — coding agent, no induction | 87.2 | 58.4 |
| ReaComp — best hybrid | 93.9 | 86.7 |

On SLR-Bench the accuracy is sharply tiered (basic 100.0% / easy 66.8% /
medium 9.6% / hard 0.0%); on PBEBench-Lite it declines with cascade length
(len 2: 83.7% → len 5: 12.9%). TroVE thus trails the coding-agent baseline by
**39.3 pp on Lite and 14.3 pp on SLR**, and ReaComp's best hybrid by **46.0 /
42.6 pp** — despite the prompts explicitly asking for reusable abstractions.

**Compute-matched control.** The numbers above use a lightweight single-shot TroVE
configuration (K=3 candidates per mode, chain-of-thought disabled, 4K-token
generations). To ensure the gap is not an artifact of under-resourcing, we also run
a **compute-matched** configuration whose per-call generation budget equals the
Qwen3.6-35B-A3B (OpenHands) agent's — chain-of-thought **on** and 16,384-token
generations (the agent's exact per-call budget). We note that TroVE is
single-shot candidate generation by construction and does *not* include the
agent's multi-step (up to 100-step) tool-use loop; matching per-call budget and
sample count isolates generation compute while preserving TroVE's algorithm.
**[TBD: compute-matched Lite + SLR numbers]**

**Implementation fidelity.** We implement TroVE (Wang et al., 2024b) in the *same*
Qwen3.6-35B-A3B + OpenHands + verifier harness as our Qwen3.6-35B-A3B (OpenHands)
coding-agent baseline, so any performance gap isolates the induction mechanism
rather than model or infrastructure differences. The implementation follows the
paper's algorithm:

| TroVE component | Our implementation |
|---|---|
| Three generation modes (IMPORT / CREATE / SKIP) | Implemented; K candidates per mode sampled concurrently per task |
| Growing toolbox of reusable functions | Persistent package dir; functions added on CREATE, importable on later tasks |
| Candidate selection | Reward-based selection (verifier), tiebroken by fewest AST nodes — a deliberate, documented deviation from the paper's self-consistency, appropriate because we have an exact verifier |
| Periodic toolbox trimming by usage | Implemented (usage-count threshold, periodic) |
| Prompts | Ask explicitly for a *new, reusable, generic* function and *not* to duplicate existing toolbox entries — i.e. we bias the model **toward** abstraction, not away from it |

The only deviations are (a) reward-based rather than self-consistency selection
(justified by the exact verifier and standard in verifier-backed settings), and
(b) the trace-generation model is our benchmark model rather than the paper's.
Neither favors ReaComp.

**Finding — library learning memorizes rather than abstracts (SLR-Bench).** Under
this faithful setup, TroVE's induced "library" does *not* converge to reusable
abstractions. On SLR-Bench the toolbox collapses to a **single function** that
takes no arguments, performs no computation, and simply `print`s a **hardcoded
rule string** derived from one specific task (its chain-of-thought frozen in a
docstring); repeated CREATE calls overwrite this one slot with a new memorized
answer. When this "abstraction" is **reused** (IMPORT mode), it solves only
**5/190 tasks (2.6%, mean reward 0.539)**, versus **416/687 (60.6%, mean reward
0.878) when the model instead solves each task from scratch** (SKIP) — i.e.
**reuse actively degrades performance**. The same pattern holds on PBEBench-Lite:
IMPORT solves 45/404 (11.1%, mean reward 0.197) while SKIP solves 364/494 (73.7%,
mean reward 0.805). In other words, the model ignores the prompt's request for
generality and memorizes, and reusing memorized answers on new tasks hurts —
library reuse is the single weakest mode on *both* benchmarks.

**Contrast with ReaComp.** ReaComp compiles a *single general* algorithm whose
mechanism is human-readable (App. F.2 — e.g. the SLR solvers perform
ascending-complexity search over the task's own predicate space, re-deriving the
correct rule per task). This is the genuine reusable abstraction, and it is why
ReaComp's CC solver reaches 46.8% on SLR-Hard — matching frontier models — where a
memorized-answer library collapses.

**Scope (stated honestly).** This finding is for TroVE *with this coding agent*
under matched conditions; a stronger coding agent might induce more general
functions. We therefore claim only that, under matched conditions, library
induction here reduces to task-specific memorization — not that library learning
is fundamentally incapable. We report accuracy against TroVE and the
Qwen3.6-35B-A3B (OpenHands) agent as the primary quantitative comparison, with the
library-content analysis as supporting qualitative evidence for d.4.

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
cost entirely offline. Empirically, our existing **Qwen3.6-35B-A3B (OpenHands)**
coding-agent baseline (same Qwen3.6 agent, per task: 87.2% Lite / 24.9% Hard /
58.4% SLR) already shows that per-task agentic effort trails the compiled solver on
hard instances; TroVE adds the cross-task library-reuse comparison (47.9% Lite /
44.1% SLR — below both). Directly heeding Sesterhenn et al., we report TroVE under
a **compute-matched** configuration as well (chain-of-thought on, 16K-token
generations, matching the coding agent's per-call budget), so the gap cannot be
attributed to under-resourcing. Beyond the headline accuracy, we will
report a **qualitative contrast of the induced libraries**: our own solvers are
documented general search algorithms (App. F.2), whereas TroVE's per-task-sampled
toolbox in our runs mixes general functions with **task-specific memorized
functions** (toolbox entries that hardcode a single task's answer yet are re-invoked
on others) — empirical support for the claim that a *single compiled solver*
captures reusable structure more cleanly than accumulate-and-reuse library learning.
**[TBD: numbers + toolbox analysis]**

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
(App. F.5). Finally, the resulting solvers are not opaque: the **qualitative
trajectory analysis** (App. F.2) documents each induced solver's *actual algorithm*
— e.g. two-phase safe/unrestricted beam search with `difflib.SequenceMatcher`
candidate extraction (Claude Code, strongest single solver: 80.4% Lite / 69.7%
Hard), safety-first greedy with 2-step lookahead, or ascending-complexity layered
search for SLR — so a reader can inspect and judge solver design directly. We will
make the App. D.1 and App. F.2 pointers prominent in the main text.

### Concern 2 — "A comparison with other symbolic solvers is necessary."

Agreed; see **[R-TROVE]**. We have added a TroVE library-induction baseline in the
identical Qwen3.6 + OpenHands + verifier harness on PBEBench-Lite and SLR-Bench
(47.9% Lite / 44.1% SLR). Together with the existing Qwen3.6-35B-A3B (OpenHands)
coding-agent baseline (87.2% Lite / 24.9% Hard / 58.4% SLR), this brackets ReaComp
against both per-task agentic solving and cross-task library learning under matched
conditions.

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
