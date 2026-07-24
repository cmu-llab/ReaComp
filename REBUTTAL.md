# ReaComp: Rebuttal / Author Response

We thank the Area Chair and all three reviewers. A few concerns come up in more
than one review (a matched baseline, generalization, and domain scope), so we
answer those once in a General Response and point to it from the individual
replies. All numbers below are verifier-exact. A couple of items are still
finishing during the rebuttal period and are marked [TBD].

---

# 1. General Response

### G1. A matched library-learning baseline (TroVE)

The most common request is a comparison against other neuro-symbolic and
library-learning methods. We added TroVE (Wang et al., 2024b, which we already
cite), run inside the same Qwen3.6-35B-A3B + OpenHands + verifier setup as our
coding-agent baseline. Keeping the harness fixed means any difference reflects the
induction method itself rather than the model or the infrastructure.

| System | PBEBench-Lite | SLR-Bench |
|---|--:|--:|
| TroVE (library induction) | 47.9% | 44.1% |
| Qwen3.6-35B-A3B (OpenHands), coding agent, no induction | 87.2% | 58.4% |
| ReaComp, best hybrid | 93.9% | 86.7% |

TroVE lands well below both: 39.3 and 14.3 points under the coding agent, and 46.0
and 42.6 points under ReaComp. Its SLR accuracy is steeply tiered (basic 100%, easy
66.8%, medium 9.6%, hard 0.0%), so it fails exactly on the hard tier where
structured search matters, whereas ReaComp's CC solver reaches 46.8% there. (TroVE
uses fewer total tokens than the coding agent because it samples a fixed set of
single-shot candidates rather than running a multi-step agentic loop; to rule out
under-resourcing we are also completing a per-call compute-matched run, reported in
the [TBD] below.)

On fidelity: we implement TroVE's three modes (IMPORT, CREATE, SKIP), a growing
importable toolbox, usage-based trimming, and candidate selection. We select by
verifier reward instead of self-consistency, which is standard when an exact
verifier is available, and the trace model is our benchmark model rather than the
paper's. Neither choice works in ReaComp's favor, and the prompts explicitly ask the
model for reusable, generic functions.

The more interesting result speaks to meta-review point (d.4). TroVE's induced
library does not become a set of reusable abstractions; it memorizes. On SLR the
toolbox collapses to a single function that takes no arguments and prints a hardcoded
rule copied from one task, with that task's chain-of-thought left in a docstring, and
each CREATE call simply overwrites it with another task's answer. When the model
reuses this function (IMPORT) it solves 5 of 190 tasks (mean reward 0.54); when it
instead solves each task from scratch (SKIP) it solves 416 of 687 (mean reward 0.88).
Reusing the "library" actually lowers performance, and the same holds on Lite (IMPORT
0.20 vs SKIP 0.81). ReaComp compiles one general algorithm whose logic is readable in
the code (App. F.2; the SLR solver, for instance, does ascending-complexity search
over each task's own predicates). We are careful about scope: this is TroVE with this
particular coding agent, and a stronger agent might do better, so we do not claim
library learning cannot work in general. [TBD: a compute-matched TroVE run with
chain-of-thought on and a 16K-token budget.]

### G2. Generalization beyond the training distribution

The paper already tests this. In the forward-reconstruction study (App. F.4), the
induced solvers are applied zero-shot to real historical-linguistics data with an
unseen IPA alphabet, variable and unknown cascade lengths, and no ground-truth
programs. That is a real distribution shift, not a resampling of the training set.
The solvers reach about 70% individually and 80.1% when ensembled, with no
retraining. One useful property of a standalone symbolic solver is that on a novel
input it cannot quietly fall back on an LLM's priors; it either generalizes on its
own terms or fails visibly. We are explicit about the boundary: this holds for
distributions that share the DSL's structure, not for arbitrary new task formats
(App. B).

### G3. Sensitivity to trace quality and induction variance

Both are measured in the paper (App. F.2, Table 13).

Trace quality: dropping chain-of-thought from the demonstrations lowers
PBEBench-Hard from 74.7% to 24.8% (and Lite from 53.4% to 42.1%), while cutting the
number of CoT examples from 100 to 48 barely moves the result. The signal comes from
the reasoning content, not the example count. Failed and noisy traces are handled by
construction: the trace set is balanced across success and failure, and the agent
picks up failure modes from it, such as the safety-first strategy that avoids editing
already-correct examples.

Induction variance: three runs on identical traces span 53.4-79.2% on Lite and
51.8-74.7% on Hard, each arriving at a different algorithm. We would add that this
variance is useful rather than only a liability. Because each run discovers a
different algorithm, the solvers have slightly different failure modes, so ensembling
them beats any single solver (91.3% Lite and 84.7% Hard, above the best individual
run in both cases). In other words, variance across induction attempts is a source of
solver diversity that the ensemble exploits. Separately, inference for a fixed solver
is near-deterministic (Table 18: 99.6% and 97.0% agreement across reruns), so the
variance is in what gets induced, not in how a given solver behaves at test time.

### G4. Domain scope and open-world benchmarks

The two-domain plus real-world scope is deliberate. Solver induction pays off
precisely where a fast, exact verifier exists, since that is what lets the agent test
and refine a solver offline. And these are not easy domains: PBEBench-Hard (cascades
up to length 20, a search space around 10^206) and SLR-Hard are where LLM scaling
collapses and ReaComp does best.

On ARC-AGI, which reviewer orXJ suggests, we think it is genuinely out of scope for
offline solver induction, for a principled reason. ReaComp compiles the reasoning of
one shared task distribution into a single reusable solver over a fixed DSL. ARC-AGI
is open-world by design: its tasks are separated so that each can require novel
primitives, and there is no DSL or task family shared across them. State of the art
work on ARC bears this out. Li et al. (2025) deliberately avoid a fixed DSL, arguing
it "restricts the class of allowed programs," and instead allow arbitrary Python to
cover ARC's long tail, reaching about 54% only after expanding 160 human seeds into
400k synthetic problems and fine-tuning an 8B model. A method built around a shared
DSL is the wrong tool there. We will scope our claims to the verifier-backed PBE and
ILP setting we actually evaluate, and list ARC-family and List-Functions benchmarks
as future work.

---

# 2. Response to the Meta-Review

We appreciate the summary and the specific conditions in point (d). Most of them are
already supported by results in the submission (mainly Appendices F.2 and F.4), which
we will move into the main text so they are easier to find.

- **(d.1) Trace-quality sensitivity:** see G3 (no-CoT drops Hard from 74.7% to 24.8%).
- **(d.2) Induction variance:** see G3 (53-79% across identical-data runs, recovered
  by ensembling).
- **(d.3) Generalization:** see G2 (zero-shot transfer to real IPA data, 80.1%).
- **(d.4) Reusable abstractions vs. task-specific heuristics:** the induced solvers
  are documented general algorithms (App. F.2), and the new TroVE baseline is the
  contrast case, where library learning memorizes instead (G1).
- **(d.5) Stronger baselines:** see G1 (TroVE in the matched harness).
- **(d.6) Broader benchmarks and verifiability:** see G4.

On the point about lacking theory: we agree the contribution is empirical. We will
frame the regularities we observe as conditions under which induction is expected to
work (the traces carry reasoning, the target shares the DSL's structure, and the
verifier is fast and exact) rather than claim more than we can show. We will also
narrow the paper's claims to the verifier-backed PBE and ILP setting.

---

# 3. Reviewer ggna (Rating: 4, Borderline Accept)

**W1/Q1 (trace quality), W2/W3 (generalization and failed traces), Q2 (baselines):**
answered in G3, G2, and G1.

**W4, Table 1 showing lower tokens but higher cost.** This is a presentation problem
and we will fix it. The token column counts only LLM fallback tokens, so tasks the
solver handles contribute zero. The cost column additionally includes the one-time
solver construction cost ($2.00 for CC, $0.85 for QO). On Lite, where accuracy is
already near ceiling, the hybrid saves inference tokens but that fixed construction
charge can make the total dollar figure match or slightly exceed BoK. This is why we
describe Lite as an efficiency regime, and why the large cost reductions show up on
Hard (78% fewer tokens). We will split the table into separate inference-cost and
amortized-construction-cost columns.

**Clarity, "DSL is not defined":** it is defined in §3 (replace(A,B) with 1<=|A|<=3
and 0<=|B|<=3, and the Prolog form eastbound(T):-Body.). The problem is placement:
the abstract symbol P appears in §2, about a page earlier. We will give the concrete
DSL at that first mention and add a short boxed definition in §3.

**Clarity, highlighting:** we will use one consistent "best per column" convention
across Tables 1-3.

---

# 4. Reviewer ZhtM (Rating: 5, Accept)

**W (domain-bound) and Q5, is the solver bound to the training domain?** Largely
yes, and the reviewer is right about the scope, with one qualifier. The solver is
tied to the DSL's structure, but not to the exact training distribution. As shown in
G2 (General Response), the solvers transfer to real historical sound-law induction at
80.1% accuracy despite never being induced on actual IPA strings from real languages,
and they recover some linguistically plausible sound laws. So the correct
characterization is that they generalize across distributions that share the DSL, not
that they only work on the training tasks themselves.

**Q1, trace model:** gpt-oss-120b, and it is the same across all runs and ablations.
It is a separate model from the coding agent that writes the solver (Claude Code with
claude-sonnet-4-6, or Qwen3.6-35B-A3B via OpenHands). We will state this distinction
clearly.

**Q2, solver form and bias:** the solver is constrained at the interface, not in its
algorithm. The prompt (App. D.1) fixes the function signature, read-only verifier
access, the DSL, and some soft preferences, but not the search procedure. The
clearest evidence that we inject little algorithmic bias is that identical inputs
produce qualitatively different algorithms across runs (Table 13).

**Q3, search size and timeout:** the solvers do not enumerate the space (around
10^206 on Hard). They extract candidates from input-output diffs and run bounded beam
or greedy search. Yes, we do cap execution time: each task runs in its own child
process and the evaluator kills it if it exceeds a per-task limit (60s for PBEBench,
3600s for SLR-Bench), marking it failed. The larger SLR-Bench limit is not for a
bigger search but for the verifier: scoring a candidate rule shells out to
SWI-Prolog (about 300ms per call), so a task that checks many candidates spends most
of its time in the verifier rather than in the solver's own search. In practice the
limit is slack rather than tight, since a typical solve finishes well under it. The
concern that a solvable task is being cut off early is reasonable, but the evidence
points the other way, and note
that if it were happening our reported accuracy would only be an underestimate.
Giving the search more room does not recover solutions: raising the maximum cascade
length from 20 to 100 adds only about a point (App. F.4). And failures are not the
search sitting idle until it is killed; they are near-misses, with reward above 0.92
even at length 20. So the binding constraint is the solver's algorithm, not the time
limit.

**Q4, qualitative example:** we will add a worked figure showing a trace, the
mechanism it inspired, and a task the resulting solver solves. App. F.4 already lists
recovered rules such as replace('ʔ','') for glottal-stop deletion.

**Q6, inducing a solver from test-task traces:** we like this idea, and in fact this
is already what ReaComp does. Solver induction is label-free: per §2.2, the coding
agent never sees any ground-truth program, only the input-output examples that are
available to any solver at test time, and the induction instances are just a small
subset of the benchmark tasks. So there is no supervised train/test distinction here.
The "training" tasks are simply unlabeled instances that supply LLM reasoning traces
(including failed attempts with partial ideas, since the trace set is balanced across
success and failure), which is exactly the regime the reviewer describes. The only
degree of freedom is which unlabeled instances provide the traces. It is also
practical, since induction needs very little data: our 12-example ablation induces a
usable solver from just 12 traces (App. F.2, Table 13). The one cost to weigh, as the
reviewer notes, is the coding agent's construction run. We will make this label-free
framing explicit in the paper.

**Suggestions:** we will tighten the candidate-set notation (C = {p_k}, and C_S,
C_L), move the ablation grid out of the main table, fix the bolding, and add
Vision-Language Programs [1] and ActivationReasoning [2] to Related Work.

[1] Wüst, A., Stammer, W., Shindo, H., Helff, L., Dhami, D. S., & Kersting, K. Synthesizing Visual Concepts as Vision-Language Programs. CVPR 2026.

[2] Helff, L., Härle, R., Stammer, W., Friedrich, F., Brack, M., Wüst, A., Shindo, H., Schramowski, P., & Kersting, K. ActivationReasoning: Logical Reasoning in Latent Activation Spaces. ICLR 2026.

---

# 5. Reviewer orXJ (Rating: 2, Reject, Confidence: 5)

**Concern 1, "generated from a simple one-paragraph prompt":** we think this is a
misreading. That sentence is the task line of a longer specification (App. D.1) that
fixes the interface, gives read-only verifier access, states the DSL, and lists
behavioral requirements. The solver is not produced in one shot; it comes out of a
multi-step agentic loop (49 to 102 turns per run; Alg. 1, Table 14). Its quality is
established by the verifier, not asserted: every solved task scores reward 1.0 under
exact execution, with per-length, per-tier, and per-BFCC breakdowns and significance
tests (App. F.5). Each solver's algorithm is also documented and can be read directly
(App. F.2). The code and agent-generated documentation for each solver are included in the supplementary material.

**Concern 2, comparison with other symbolic solvers:** added, see G1. ReaComp is now
placed against both per-task agentic solving and cross-task library learning in the
same harness.

**Concern 3, making better use of the solver's output in the fallback:** a fair
point. The solver already returns its top-K scored near-misses, but today the
fallback ignores them: it is invoked independently and we simply pick the best
answer by reward. Warm-starting the fallback from those near-misses, i.e. having
the LLM refine the solver's partial output rather than start from scratch, is a
natural extension, and promising because solver failures are mostly near-misses
(reward 0.987 on Hard). [TBD: a solver-seeded fallback ablation, otherwise Future
Work.]

**On the strength and framing of the contribution (the "MDP [1] / latent-reasoning
[2]" point).** We would gently reframe what ReaComp is claiming. The contribution is
not a symbolic solver hand-built for two particular domains, nor a prompt that
refines solver outputs with an LLM. It is a demonstration that end-to-end, automatic
induction of novel symbolic solvers is feasible for domains that admit a constrained
DSL and a fast, exact verifier, done entirely from LLM reasoning traces and without
any ground-truth programs (§2.2). The finding is that this is possible at all, and
that the induced solvers are both cheap (they run at zero per-task LLM cost, §2.3)
and strong (they outperform LLM test-time scaling on hard instances, Tables 2-3).
Combined with an LLM in the hybrid setting, they are Pareto-optimal on cost versus
accuracy and set the best results we are aware of on the domains we study. We read
this as a methodological result about a class of problems, DSL-complete domains with
perfect verifiers, rather than a solver for a specific benchmark. We would also
gently correct one point: the LLM does not refine the solver's output; it is invoked
only as an independent fallback on the residual tasks the solver does not resolve
(§2.3).

On more advanced formulations like [1] and [2]: we agree they are worth exploring. 
But we would gently note that "more complex" and "stronger contribution" are not the same thing. 
What ReaComp demonstrates, that novel symbolic solvers can be induced by coding agents from unlabeled tasks augmented with LLM
reasoning traces, for DSL-complete and verifiable domains, is to our knowledge
genuinely new, and the fact that it can be done this simply is part of the result
rather than a shortcoming. A simpler method that induces cheap, standalone solvers
which outperform test-time scaling might not be weaker than a more elaborate one; we
would argue the simplicity is a merit given what it achieves.

**ARC-AGI and other benchmarks [3]:** see G4. In short, ARC-AGI lacks the shared
cross-task structure that offline solver induction relies on, so we treat it as out
of scope and add the ARC-family and List-Functions benchmarks to future work.

We hope the corrected account of how the solver is built (Concern 1) and the new
matched baseline (Concern 2) address the main reservations, and we would be glad to
run additional analyses during discussion.

[1] ARCLE: The Abstraction and Reasoning Corpus Learning Environment for Reinforcement Learning, 2024 

[2] Generative Recursive Reasoning, 2026 

[3] Phenomenal Yet Puzzling: Testing Inductive Reasoning Capabilities of Language Models with Hypothesis Refinement, 2023

---

## Summary of paper changes

1. Add the TroVE baseline (matched harness) and the library-content comparison for
   (d.4).
2. Give the concrete DSL at the first mention of P in §2 and add a boxed definition
   in §3.
3. Split Table 1 cost into inference cost vs. amortized construction cost.
4. Use consistent bolding across Tables 1-3 and move the ablation grid to its own
   table.
5. Tighten the candidate-set notation and add a worked trace-to-solver-to-task
   example.
6. Add a Limitations paragraph on trace quality, and clarify the trace model vs. the
   coding agent, the interface-vs-algorithm constraint, and the execution bounds.
7. Add related work and positioning: Vision-Language Programs, ActivationReasoning,
   ARCLE, latent reasoning, and Li et al. (2025) on ARC.
8. Narrow the claims to the verifier-backed PBE/ILP setting, with ARC-family
   benchmarks, solver-seeded fallback, and per-instance induction as future work.
