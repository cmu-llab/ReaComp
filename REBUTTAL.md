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
| TroVE (compute-matched: CoT, 16K tokens) | 53.2% | 54.5% |
| Qwen3.6-35B-A3B (OpenHands), coding agent, no induction | 87.2% | 58.4% |
| ReaComp, best hybrid | 93.9% | 86.7% |

TroVE lands well below both: 39.3 and 14.3 points under the coding agent, and 46.0
and 42.6 points under ReaComp. Its SLR accuracy is steeply tiered (basic 100%, easy
66.8%, medium 9.6%, hard 0.0%), so it fails exactly on the hard tier where
structured search matters, whereas ReaComp's CC solver reaches 46.8% there. (TroVE
uses fewer total tokens than the coding agent because it samples a fixed set of
single-shot candidates rather than running a multi-step agentic loop; to rule out
under-resourcing we also ran a per-call compute-matched configuration, reported
below.)

On fidelity: we implement TroVE's three modes (IMPORT, CREATE, SKIP), a growing
importable toolbox, usage-based trimming, and candidate selection. We select by
verifier reward instead of self-consistency, which is standard when an exact
verifier is available, and the trace model is our benchmark model rather than the
paper's. Neither choice works in ReaComp's favor, and the prompts explicitly ask the
model for reusable, generic functions.

The more interesting result speaks to meta-review point (d.4). TroVE's induced
library does not become a set of reusable abstractions; it memorizes. On SLR
(compute-matched run) the toolbox is a handful of near-duplicate functions
(`find_eastbound_rule`, `find_train_rule`, `find_train_eastbound_rule`, ...), each
taking no arguments and specialized to one task's data, with that task's specific
train identifiers baked into the body. The decisive evidence is in how these get
used. When the model imports a toolbox function (IMPORT) it solves 2 of 393 tasks
(1%, mean reward 0.52); when it instead writes a solution for the task in front of it
(SKIP) it solves 538 of 584 (92%, mean reward 0.98). Reusing the "library" does not
help, it hurts, and the same pattern holds on Lite (IMPORT 24% vs SKIP 67%). In other
words the accumulated library is close to dead weight: TroVE succeeds when it ignores
it. This is consistent with the compute-matched re-evaluation of TroVE on MATH by
Sesterhenn et al. (2025), which we cite, and which independently finds TroVE's tools
to be trivial or rarely reused and its apparent gains to come from extra compute
rather than reuse (shrinking to about 1% once compute is matched). Our two findings,
that matched compute closes most of TroVE's headroom and that reuse itself does not
help, point the same way from a different domain. ReaComp instead compiles one general
algorithm whose logic is readable in the code (App. F.2; the SLR solver, for instance,
does ascending-complexity search over each task's own predicates). We are careful
about scope: this is TroVE with this particular coding agent, and a stronger agent
might do better, so we do not claim library learning cannot work in general.

The per-call compute-matched run (chain-of-thought on, 16K-token budget) confirms the
gap is not an artifact of under-resourcing. It raises TroVE to 53.2% on Lite (from
47.9%) and 54.5% on SLR (from 44.1%, with tiers basic 100%, easy 76.4%, medium 29.6%,
hard 12.0%). The extra budget helps, most visibly lifting the SLR hard tier off the
floor (0.0% to 12.0%), but library induction under matched compute still trails the
coding agent (87.2% / 58.4%) and ReaComp's best hybrid (93.9% / 86.7%) by wide
margins on both benchmarks.

### G2. Generalization beyond the training distribution

The paper already tests this in the main text. The real-world forward-reconstruction
case study (§4, "Real-world case study") applies the induced solvers zero-shot to real
historical-linguistics data with an unseen IPA alphabet, variable and unknown cascade
lengths, and no ground-truth programs, reaching about 70% individually and 80.1%
ensembled with no retraining (§5 summarizes the finding; full setup and qualitative
examples in App. F.4). That is a real distribution shift, not a resampling of the
training set. One useful property of a standalone symbolic solver is that on a novel
input it cannot quietly fall back on an LLM's priors; it either generalizes on its own
terms or fails visibly. We are explicit about the boundary: this holds for
distributions that share the DSL's structure, not for arbitrary new task formats.

### G3. Sensitivity to trace quality and induction variance

Both are measured in the paper. The solver-induction ablations are reported in §4
("Solver induction ablations") and discussed in §5, with the full grid in App. F.2
(Table 13).

Trace quality: dropping chain-of-thought from the demonstrations lowers
PBEBench-Hard from 74.7% to 24.8% (and Lite from 53.4% to 42.1%), while cutting the
number of CoT examples from 100 to 48 barely moves the result. The signal comes from
the reasoning content, not the example count. Failed and noisy traces are handled by
construction: the trace set is balanced across success and failure, and the agent
picks up failure modes from it, such as the safety-first strategy that avoids editing
already-correct examples.

Induction variance: three runs on identical traces span 53.4-79.2% on Lite and
51.8-74.7% on Hard, each arriving at a different algorithm. We would add that this
variance is useful, because each run discovers a different algorithm. 
The solvers have slightly different failure modes, so ensembling
them beats any single solver (91.3% Lite and 84.7% Hard, above the best individual
run in both cases). In other words, variance across induction attempts is a source of
solver diversity that the ensemble exploits. Separately, inference for a fixed solver
is near-deterministic (Table 18: 99.6% and 97.0% agreement across reruns), so the
variance is in what gets induced, not in how a given solver behaves at test time.

### G4. Domain scope and open-world benchmarks

The two-domain plus real historical linguistics scope is deliberate. Solver induction pays off
precisely where a fast, exact verifier exists, since that is what lets the agent test
and refine a solver offline. And these are not easy domains: PBEBench-Hard (cascades
up to length 20, a search space around 10^206) and SLR-Hard are where LLM scaling
collapses and ReaComp does best.

ARC (raised by reviewer orXJ), draws a useful boundary around what ReaComp assumes. 
ReaComp compiles the reasoning of one shared task distribution into a single reusable solver
over a fixed DSL, so it fits domains that are effectively DSL-complete: a compact
shared vocabulary covers the task family, and a fast exact verifier is available. 
ARC was designed to violate exactly this assumption. 
Its tasks are deliberately separated so that each can call for novel primitives, and there is no single DSL or task family
shared across them.

The ARC literature reflects this boundary. DSL-based program search was central to
early ARC progress and remains competitive: the 2020 Kaggle winner (Wind, 2020) was a
hand-built DSL of grid operations with brute-force search, and Hodel's arc-dsl is a
widely used reference DSL. What has changed is that the strongest recent approaches
generalize the search space beyond any fixed DSL. Li et al. (2024) keep a shared
primitive library but "still allow arbitrary Python code, which helps cover the long
tail of diverse tasks," reaching about 54% by expanding roughly 160 human seeds into
400k synthetic problems; others rely on test-time training. 
Notably, ever since ARC-AGI-2 the data has been constructed to remove the tasks
solvable by DSL-searching approaches, which suggests that a fixed DSL alone plateaus on
ARC.

However, we still read this as a promising next step.
Extending induction to domains that are not DSL-complete, ARC among them, is promising future work, and we
suspect a fixed DSL will not suffice on its own there but could still serve as a useful backbone inside a broader arbitrary-program synthesizer (much as Li et al.'s shared library sits under arbitrary Python). 
For the current paper we will scope our claims to the verifier-backed PBE and SLR setting we actually evaluate, and list the ARC-family, MiniSCAN, and List Functions benchmarks as future work (with a List Functions pilot reported in our reply to orXJ below).

---

# 2. Response to the Meta-Review

We thank the Area Chair for the summary and for the specific conditions in point (d).
We address each below and point to where the evidence lives in the paper.

- **(d.1) Trace-quality sensitivity:** §4 ("Solver induction ablations") reports that
  removing chain-of-thought drops PBEBench-Hard from 74.7% to 24.8%, and §5 draws out
  the implication; the full ablation grid is in App. F.2. For more details see G3 in the general response.
- **(d.2) Induction variance:** §4 (same paragraph) reports 53.4-79.2% on Lite and
  51.8-74.7% on Hard across runs on identical data, and §5 frames induction as a search
  over algorithms rather than a data-scaling problem. Ensembling recovers the spread
  (details in G3 in the general response).
- **(d.3) Generalization:** §4 ("Real-world case study") reports zero-shot transfer to
  real IPA data at up to 80.1% by union, and §5 discusses it; setup and qualitative
  examples are in App. F.4. See G2.
- **(d.4) Reusable abstractions vs. task-specific heuristics:** the induced solvers run
  documented general algorithms (the SLR solver's ascending-complexity search is named
  in §4), and the new TroVE baseline is the contrast case, where library learning
  memorizes instead (G1). Per-solver mechanism analysis is in App. F.2.
- **(d.5) Stronger baselines:** see G1 from the genral response (TroVE in the matched harness).
- **(d.6) Broader benchmarks and verifiability:** see G4 from the general response.

On the point about lacking theory: we agree the contribution is empirical. We will
frame the regularities we observe as conditions under which induction is expected to
work (the traces carry reasoning, the target shares the DSL's structure, and the
verifier is fast and exact) rather than claim more than we can show. We will also
narrow the paper's claims to the verifier-backed PBE and SLR setting.

The two items that currently live only in the appendix are the per-solver mechanism
analysis (App. F.2) and the generalization setup and qualitative examples (App. F.4).
For the revision we will move these into the main text so the body carries the
argument on its own, keeping the appendix for the fuller breakdowns.

---

# 3. Reviewer ggna (Rating: 4, Borderline Accept)

**W1/Q1 (trace quality), W2/W3 (generalization and failed traces), Q2 (baselines):**
answered in General response (G3, G2, and G1).

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
across Tables 1-3, bolding the best value in each column across all rows (reported
baselines included, not only our own methods). On the primary accuracy metric a
ReaComp method is the best in every table and every SLR tier. On the efficiency
columns (tokens, cost) the standalone symbolic solvers are best, and on the hard
splits the symbolic solvers lead on mean reward and edit similarity while the hybrids
lead on accuracy, so the column-best is a ReaComp method throughout even though it is
not always the hybrid.

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
more clearly.

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
reviewer notes, is the coding agent's construction run. We will make this **label-free**
framing explicit in the paper.

**Suggestions:** we will tighten the candidate-set notation (C = {p_k}, and C_S,
C_L), move the ablation grid out of the main table, fix the bolding, and add
Vision-Language Programs [1] and ActivationReasoning [2] to Related Work.

[1] Wüst, A., Stammer, W., Shindo, H., Helff, L., Dhami, D. S., & Kersting, K. Synthesizing Visual Concepts as Vision-Language Programs. CVPR 2026.

[2] Helff, L., Härle, R., Stammer, W., Friedrich, F., Brack, M., Wüst, A., Shindo, H., Schramowski, P., & Kersting, K. ActivationReasoning: Logical Reasoning in Latent Activation Spaces. ICLR 2026.

---

# 5. Reviewer orXJ (Rating: 2, Reject, Confidence: 5)

**Concern 1, "generated from a simple one-paragraph prompt":** the solvers are not
produced from a one-paragraph prompt. The paragraph the reviewer quotes is the task
line of a longer specification (App. D.1) that also fixes the function interface, grants
read-only verifier access, states the DSL, and lists behavioral requirements, and the
solver is built through a multi-step agentic loop of 49 to 102 turns per run in which
the agent proposes, executes, and revises code against the verifier. That is the
process the paper evaluates, not a single generation. We accept that the body did not
make this visible enough, since the loop and the interface constraints sit in the
appendix (Alg. 1 and Table 14), and we will try to move them into the main method section so
this reads correctly without the appendix. Solver quality is then established by the
verifier, not asserted: every solved task scores reward 1.0 under exact execution, and
we report per-length, per-tier, and per-BFCC breakdowns with significance tests. The
induced solvers are also legible as algorithms rather than opaque blobs, and they
arrive at genuinely different strategies. For PBEBench, one run learns a safety-first
greedy search with a hard constraint against modifying already-correct examples plus
2-step lookahead for interaction effects; another separates forced from optional edit
operations and enumerates permutations over the forced ones before greedy search;
another runs a two-phase beam search with candidates extracted from
difflib.SequenceMatcher, where a safe phase precedes an unrestricted one. For SLR the
solver searches in ascending complexity layers, trying one-literal rules, then two, and
so on until the verifier accepts. These are strategies a human synthesis expert would
recognize, not a single memorized answer, and the code and agent-generated
documentation for every solver are in the supplementary material (App. F.2) so they can
be read directly.

**Concern 2, comparison with other symbolic solvers:** added, see G1. ReaComp is now
placed against both per-task agentic solving and cross-task library learning in the
same harness.

**Concern 3, making better use of the solver's output in the fallback:** a fair
point. The solver already returns its top-K scored near-misses, but right now the
fallback ignores them: it is invoked independently and we simply pick the best
answer by reward. Warm-starting the fallback from those near-misses, i.e. having
the LLM refine the solver's partial output rather than start from scratch, is a
natural extension, and promising because solver failures are mostly near-misses
(reward 0.987 on Hard). [TBD: a solver-seeded fallback ablation, otherwise Future
Work.]

**On the strength and framing of the contribution (the "MDP [1] / latent-reasoning [2]" point).** The reviewer is right that these are the relevant lines of work to
situate against, and we should have positioned ReaComp among them explicitly. Doing so
also clarifies what the contribution is: not a hand-built solver for two domains, but a
demonstration that novel symbolic solvers can be induced automatically, from LLM
reasoning traces and without ground-truth programs (§2.2), for domains that admit a
constrained DSL and a fast exact verifier.

The MDP formulation (ARCLE [1]) casts solving as a sequence of grid-edit actions
learned with RL. ReaComp keeps the idea of search over a structured action space but
moves it offline: the coding agent searches over whole programs, scored by a verifier,
and compiles the result into a standalone solver that then runs the task distribution
at zero per-task cost. This is a real design difference, and ARCLE's own finding that
RL over the raw action space is hard supports the point that the leverage comes from
verifier-scored program-level search rather than step-by-step exploration. Where a
verifier is not available, the RL formulation has the advantage, and we do not claim
otherwise.

The latent-reasoning line (GRAM [2] and the recursive-reasoning models it builds on)
amortizes reasoning into a latent trajectory and predicts the answer directly. That
buys a smooth, trainable representation and needs no DSL, which is a genuine strength on
open-world tasks. ReaComp makes the opposite trade: it externalizes reasoning as an
explicit program, giving up differentiability but gaining an artifact that is
inspectable, reusable, and basically free to run at test time. 
Our results show that for DSL-complete, verifiable domains this trade pays off, the induced solvers outperform LLM test-time scaling on hard instances (Tables 2-3) at zero per-task LLM cost. 
We read the two as complementary rather than competing, and combining an induced symbolic backbone with
latent reasoning on the residual is a direction worth pursuing.

We would push back on one framing, that a more complex formulation is automatically a
stronger contribution. What ReaComp shows is new to our knowledge, and that it works
with a simple recipe is part of the finding: it sets a concrete baseline that richer
MDP or latent-reasoning methods can be measured against on these domains. We do not
offer simplicity as a substitute for those formulations, only as evidence of how far
verifier-backed induction with a coding agent and LLM reasoning traces already gets. 
One factual clarification: in the hybrid setting the LLM does not refine the solver's output, it is an independent fallback on the tasks the solver leaves unresolved (§2.3); we will make this clearer in the text.

**ARC-AGI and other benchmarks [3]:** see G4 for the ARC-specific discussion. On the
broader list the reviewer raises (Mini-ARC, ARC-AGI-1/2, MiniSCAN, List Functions,
ACRE), it is worth separating them by whether they fit the setting ReaComp assumes, a
symbolic input-to-output transformation with a fast exact verifier. List Functions and
MiniSCAN do: both are symbolic (integer-list to integer-list, and compositional
word-to-symbol-sequence) with exact-match verifiers and compact primitives. The
ARC-family is verifiable too (grid equality is exact) but is deliberately not
DSL-complete, which is the boundary we discuss in G4. ACRE is the one genuine mismatch:
it is a visual abductive-causal benchmark (CLEVR-style rendered scenes with a Blicket
detector) that requires perception and causal inference rather than a string-to-string
transformation, and it has no exact program verifier of the kind we rely on, so it sits
outside the setting regardless of DSL-completeness.

To put weight behind this rather than argue it on paper, we ran a pilot of ReaComp on
List Functions during the rebuttal period: we induce a solver from LLM reasoning traces
on a small set of tasks and evaluate it on held-out inputs of the same functions.
[TBD: List Functions pilot result, with the induced solver's strategy.] We were not
able to extend this to MiniSCAN and Mini-ARC within the rebuttal window, and we will add
them, along with the full 250-task List Functions evaluation, in revision.

We would also note that [3] is in fact convergent with our approach: it finds that LLMs
are strong hypothesis proposers but weak rule appliers, and that pairing them with a
symbolic interpreter that applies and filters rules is what works, which is the same
division of labor ReaComp automates by compiling the reasoning into an executable
solver.

We hope the corrected account of how the solver is built (Concern 1) and the new
matched baseline (Concern 2) address the main reservations, and we would be glad to
run additional analyses during discussion.

[1] ARCLE: The Abstraction and Reasoning Corpus Learning Environment for Reinforcement Learning, 2024 

[2] Generative Recursive Reasoning, 2026 

[3] Phenomenal Yet Puzzling: Testing Inductive Reasoning Capabilities of Language Models with Hypothesis Refinement, 2023

[4] Li et al. Combining Induction and Transduction for Abstract Reasoning, 2024 (BARC; arXiv 2411.02272).

[5] Wind. 1st place solution, 2020 ARC Kaggle challenge, 2020. Hodel, arc-dsl (github.com/michaelhodel/arc-dsl).

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
   ARCLE, latent reasoning, and Li et al. (2024) on ARC.
8. Narrow the claims to the verifier-backed PBE/SLR setting, with a List Functions
   pilot in the rebuttal and the ARC-family and MiniSCAN benchmarks, solver-seeded
   fallback, and per-instance induction as future work.
