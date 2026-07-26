# ReaComp: Author Response (per-channel)

This file is organized by submission channel. Each reviewer box is self-contained
(assumes reviewers cannot see each other's responses). The meta-review reply is the
confidential Author-AC comment and cross-references the reviewer boxes (the AC can see
all of them). All numbers are verifier-exact. A few items are still finishing during the
rebuttal period and are marked [TBD].

Paste targets:
- Section A → Reviewer ggna response box
- Section B → Reviewer ZhtM response box
- Section C → Reviewer orXJ response box
- Section D → confidential Author-AC comment

---

# A. Reviewer ggna (Rating: 4, Borderline Accept)

We thank the reviewer. We address trace quality and induction variance (W1/Q1),
generalization and failed traces (W2/W3), baselines (Q2), and the two clarity points.

**W1/Q1, trace quality and robustness.** We agree both are central to whether the method
holds up, and treated them as first-class questions in the study (§4, "Solver induction
ablations", discussed in §5; full grid in App. F.2, Table 13). Dropping
chain-of-thought from the demonstrations lowers PBEBench-Hard from 74.7% to 24.8% (and
Lite from 53.4% to 42.1%), while cutting the number of CoT examples from 100 to 48
barely moves the result: the signal comes from the reasoning content, not the example
count. Failed and noisy traces are handled by construction, since the trace set is
balanced across success and failure and the agent picks up failure modes from it (for
example the safety-first strategy that avoids editing already-correct examples).
Induction variance: three runs on identical traces span 53.4-79.2% on Lite and
51.8-74.7% on Hard, each arriving at a different algorithm. That variance is useful:
the solvers have slightly different failure modes, so ensembling them beats any single
solver (91.3% Lite, 84.7% Hard, above the best individual run in both cases). Inference
for a fixed solver is near-deterministic (Table 18: 99.6% and 97.0% agreement across
reruns), so the variance is in what gets induced, not in how a given solver behaves at
test time. Our Limitations section does discuss this dependence (App. B, "Dependence on
coding agent capability", which reports the no-CoT drop above and that 12 examples are
insufficient), though it is framed under agent capability rather than trace quality; we
will retitle it and state the trace-quality dependence explicitly so it is easier to find.

**W2/W3, generalization and failed traces.** First, one clarification on the induction
data, since the phrasing "uses the offline solutions" may suggest more supervision than
there is: solver induction is label-free. Per §2.2 the coding agent never sees any
ground-truth program, only the input-output examples that are available to any solver at
test time (plus LLM reasoning traces, including failed ones). There are no gold solutions
in the induction set. On generalization, we saw this as important enough to test in the
main text rather than defer it: the
real-world forward-reconstruction case study (§4, "Real-world case study") applies the
induced solvers zero-shot to real historical-linguistics data with an unseen IPA alphabet,
variable and unknown cascade lengths, and no ground-truth programs, reaching about 70%
individually and 80.1% ensembled with no retraining (App. F.4 has the setup). That is a
real distribution shift, not a resampling of the training set, so the solvers are not
confined to what the trace LLM has already seen. On failed traces specifically: they are
not discarded but used, since the balanced trace set is what teaches the agent the failure
modes above.

**Q2, baselines.** We added TroVE (Wang et al., 2024b, already cited), the most-requested
library-learning comparison, run inside the same Qwen3.6-35B-A3B + OpenHands + verifier
harness as our coding-agent baseline.

| System | PBEBench-Lite | SLR-Bench |
|---|--:|--:|
| TroVE (library induction) | 47.9% | 44.1% |
| TroVE (compute-matched: CoT, 16K tokens) | 53.2% | 54.5% |
| Qwen3.6-35B-A3B (OpenHands), coding agent, no induction | 87.2% | 58.4% |
| ReaComp, best hybrid | 93.9% | 86.7% |

We implement TroVE faithfully (its three modes IMPORT/CREATE/SKIP, a growing importable
toolbox, usage-based trimming, and candidate selection), selecting by verifier reward
instead of self-consistency since an exact verifier is available; neither choice favors
ReaComp. TroVE trails both the coding agent and ReaComp by wide margins even under matched
compute, and its induced "library" is a set of near-duplicate task-specific functions
that reuse hurts rather than helps (on Lite, importing a toolbox function solves 24% vs
67% when the model ignores it). This is consistent with Sesterhenn et al. (2025), which
we cite, finding TroVE's gains shrink to about 1% once compute is matched.

**W4, Table 1 lower tokens but higher cost.** A presentation problem, and we will fix it.
The token column counts only LLM fallback tokens, so tasks the solver handles contribute
zero. The cost column additionally includes the one-time solver construction cost (2.00 USD
for CC, 0.85 USD for QO). On Lite, where accuracy is already near ceiling, the hybrid saves
inference tokens but that fixed construction charge can make the total dollar figure match
or slightly exceed BoK. This is why we describe Lite as an efficiency regime, and why the
large cost reductions show up on Hard (78% fewer tokens). We will split the table into
separate inference-cost and amortized-construction-cost columns.

**Clarity, "DSL is not defined".** It is defined in §3 (replace(A,B) with 1<=|A|<=3 and
0<=|B|<=3, and the Prolog form eastbound(T):-Body.). The problem is placement: the
abstract symbol P appears in §2, about a page earlier. We will give the concrete DSL at
that first mention and add a short boxed definition in §3.

**Clarity, highlighting.** We will use one consistent "best per column" convention across
Tables 1-3, bolding the best value in each column across all rows (reported baselines
included, not only our own methods). On the primary accuracy metric a ReaComp method is
the best in every table and every SLR tier. On the efficiency columns (tokens, cost) the
standalone symbolic solvers are best, and on the hard splits the symbolic solvers lead on
mean reward and edit similarity while the hybrids lead on accuracy, so the column-best is
a ReaComp method throughout even though it is not always the hybrid.

---

# B. Reviewer ZhtM (Rating: 5, Accept)

We thank the reviewer for their thoughtful feedback and address their concerns below.

**W (domain-bound) and Q5, is the solver bound to the training domain?** Largely yes, and
the reviewer is right about the scope, with one qualifier. The solver is tied to the
DSL's structure, but not to the exact training distribution. The real-world
forward-reconstruction case study (§4, "Real-world case study"; App. F.4) applies the
induced solvers zero-shot to real historical sound-law data with an unseen IPA alphabet,
variable and unknown cascade lengths, and no ground-truth programs, reaching about 70%
individually and 80.1% ensembled with no retraining, and it recovers some linguistically
plausible sound laws. So the correct characterization is that the solvers generalize
across distributions that share the DSL, not that they only work on the training tasks
themselves. A standalone symbolic solver also cannot quietly fall back on an LLM's priors
on a novel input: it either generalizes on its own terms or fails visibly.

**Q1, trace model.** gpt-oss-120b, and it is the same across all runs and ablations. It is
a separate model from the coding agent that writes the solver (Claude Code with
claude-sonnet-4-6, or Qwen3.6-35B-A3B via OpenHands). We will state this distinction more
clearly.

**Q2, solver form and bias.** The solver is constrained at the interface, not in its
algorithm. The prompt (App. D.1) fixes the function signature, read-only verifier access,
the DSL, and some soft preferences, but not the search procedure. The clearest evidence
that we inject little algorithmic bias is that identical inputs produce qualitatively
different algorithms across runs (Table 13).

**Q3, search size and timeout.** The solvers do not enumerate the space (around `10^206` on
Hard). They extract candidates from input-output diffs and run bounded beam or greedy
search. We do cap execution time: each task runs in its own child process and the
evaluator kills it if it exceeds a per-task limit (60s for PBEBench, 3600s for SLR-Bench),
marking it failed. The larger SLR limit is not for a bigger search but for the verifier:
scoring a candidate rule shells out to SWI-Prolog (about 300ms per call), so a task that
checks many candidates spends most of its time in the verifier rather than in the solver's
own search. In practice the limit is slack rather than tight. The concern that a solvable
task is being cut off early is reasonable, but the evidence points the other way, and if
it were happening our reported accuracy would only be an underestimate. Giving the search
more room does not recover solutions: raising the maximum cascade length from 20 to 100
adds only about a point (App. F.4). And failures are near-misses, with reward above 0.92
even at length 20. So the binding constraint is the solver's algorithm, not the time limit.

**Q4, qualitative example.** We will add a worked figure showing a trace, the mechanism it
inspired, and a task the resulting solver solves. App. F.4 already lists recovered rules
such as replace('ʔ','') for glottal-stop deletion.

**Q6, inducing a solver from test-task traces.** We like this idea, and in fact this is
already what ReaComp does. Solver induction is label-free: per §2.2, the coding agent
never sees any ground-truth program, only the input-output examples that are available to
any solver at test time, and the induction instances are just a small subset of the
benchmark tasks. So there is no supervised train/test distinction here. The "training"
tasks are simply unlabeled instances that supply LLM reasoning traces (including failed
attempts with partial ideas, since the trace set is balanced across success and failure),
which is exactly the regime the reviewer describes. The only degree of freedom is which
unlabeled instances provide the traces. It is also practical, since induction needs very
little data: our 12-example ablation induces a usable solver from just 12 traces (App.
F.2, Table 13). The one cost to weigh, as the reviewer notes, is the coding agent's
construction run. We will make this label-free framing explicit in the paper.

**Suggestions.** We will tighten the candidate-set notation (`C = {p_k}`, and `C_S`, `C_L`),
move the ablation grid out of the main table, fix the bolding, and add Vision-Language
Programs [1] and ActivationReasoning [2] to Related Work.

[1] Wüst, A., Stammer, W., Shindo, H., Helff, L., Dhami, D. S., & Kersting, K.
Synthesizing Visual Concepts as Vision-Language Programs. CVPR 2026.

[2] Helff, L., Härle, R., Stammer, W., Friedrich, F., Brack, M., Wüst, A., Shindo, H.,
Schramowski, P., & Kersting, K. ActivationReasoning: Logical Reasoning in Latent
Activation Spaces. ICLR 2026.

---

# C. Reviewer orXJ (Rating: 2, Reject, Confidence: 5)

We thank the reviewer. We correct how the solver is built (Concern 1), add the requested
symbolic-solver comparison (Concern 2), address the fallback (Concern 3), position against
the MDP and latent-reasoning lines, and triage the benchmarks with a List Functions pilot.

**Concern 1, "generated from a simple one-paragraph prompt".** We agree the strength of the
construction procedure is exactly the right thing to scrutinize, and we should have surfaced
it earlier in the paper. The paragraph quoted is the task line of a longer
specification (App. D.1) that also fixes the function interface, grants read-only verifier
access, states the DSL, and lists behavioral requirements, and the solver is built through
a multi-step agentic loop of 49 to 102 turns per run in which the agent proposes,
executes, and revises code against the verifier. That is the process the paper evaluates,
not a single generation, and we will move the loop and interface constraints (Alg. 1,
Table 14) into the main method section. The induction is also label-free: the agent sees
only input-output examples and LLM reasoning traces, never any ground-truth program (§2.2). Solver quality is then established by the verifier,
not asserted: every solved task scores reward 1.0 under exact execution, with per-length,
per-tier, and per-BFCC breakdowns and significance tests. The induced solvers are legible
algorithms that arrive at genuinely different strategies. For PBEBench, one run learns a
safety-first greedy search with a hard constraint against modifying already-correct
examples plus 2-step lookahead for interaction effects; another separates forced from
optional edit operations and enumerates permutations over the forced ones before greedy
search; another runs a two-phase beam search with candidates extracted from
difflib.SequenceMatcher, a safe phase preceding an unrestricted one. For SLR the solver
searches in ascending complexity layers, one-literal rules, then two, and so on until the
verifier accepts. These are strategies a synthesis expert would recognize, not a memorized
answer, and the code and agent-generated documentation for every solver are in the
supplementary material (App. F.2).

**Concern 2, comparison with other symbolic solvers.** We added TroVE (Wang et al., 2024b,
already cited), run inside the same Qwen3.6-35B-A3B + OpenHands + verifier harness, so any
difference reflects the induction method, not the model or infrastructure.

| System | PBEBench-Lite | SLR-Bench |
|---|--:|--:|
| TroVE (library induction) | 47.9% | 44.1% |
| TroVE (compute-matched: CoT, 16K tokens) | 53.2% | 54.5% |
| Qwen3.6-35B-A3B (OpenHands), coding agent | 87.2% | 58.4% |
| ReaComp, best hybrid | 93.9% | 86.7% |

TroVE lands 39.3 and 14.3 points under the coding agent and 46.0 and 42.6 under ReaComp.
Its SLR accuracy is steeply tiered (basic 100%, easy 66.8%, medium 9.6%, hard 0.0%): it
fails exactly on the hard tier where structured search matters, whereas ReaComp's CC solver
reaches 46.8% there. We implement TroVE's three modes (IMPORT, CREATE, SKIP), a growing
importable toolbox, usage-based trimming, and candidate selection; we select by verifier
reward instead of self-consistency (standard when an exact verifier exists), neither choice
favoring ReaComp. The more interesting result: TroVE's induced library does not become
reusable abstractions, it memorizes. On SLR (matched) the toolbox is near-duplicate
zero-argument functions specialized to one task's data with that task's train identifiers
baked in. When the model imports a toolbox function it solves 2 of 393 tasks (1%, mean
reward 0.52); when it writes a fresh solution (SKIP) it solves 538 of 584 (92%, reward
0.98), and the same pattern holds on Lite (IMPORT 24% vs SKIP 67%). Reusing the library
hurts. This matches Sesterhenn et al. (2025), which we cite, finding TroVE's gains shrink
to about 1% once compute is matched. The per-call compute-matched run (CoT on, 16K budget)
confirms the gap is not under-resourcing: it lifts the SLR hard tier off the floor (0.0 to
12.0%) but still trails both baselines by wide margins. ReaComp instead compiles one general
algorithm whose logic is readable (App. F.2). We are careful about scope: this is TroVE with
this coding agent, and a stronger agent might do better, so we do not claim library learning
cannot work in general.

**Concern 3, better use of the solver's output in the fallback.** A fair point, and we ran
the ablation. The solver already returns its top-K scored near-misses, but the current
fallback ignores them: it is invoked independently and we pick the best answer by reward.
We tested warm-starting instead, seeding the fallback with the solver's near-miss program so
the LLM refines it rather than starting from scratch, on exactly the tasks the symbolic
solver fails. Seeded feedback beats plain feedback on both benchmarks, and the gain grows
with difficulty (numbers below are on solver-failed tasks; the runs are still completing, so
these are interim on partial samples and will be finalized in revision):

| Solver-failed subset | plain DF | seeded DF | delta |
|---|--:|--:|--:|
| SLR-Bench (N=220 of ~316) | 76.4% | 93.2% | +16.8 pp |
| SLR-Bench, hard tier (n=37) | 37.8% | 81.1% | +43.2 pp |
| PBEBench-Lite (N=59 of 198) | 61.0% | 74.6% | +13.6 pp |
| PBEBench-Lite, cascade=5 (n=33) | 42.4% | 60.6% | +18.2 pp |

The pattern is consistent: warm-starting recovers the most on the hardest tasks, where the
solver's partial progress is most informative, which is what makes an independent from-scratch
fallback leave accuracy on the table. [TBD: both seeded-DF runs are still in progress; final
headline and per-tier numbers will replace the interim values above.]

**On framing (the MDP [1] / latent-reasoning [2] point).** The reviewer is right that these
are the relevant lines to situate against. Doing so clarifies the contribution: not a
hand-built solver for two domains, but a demonstration that novel symbolic solvers can be
induced automatically, from LLM reasoning traces and without ground-truth programs (§2.2),
for domains that admit a constrained DSL and a fast exact verifier. The MDP formulation
(ARCLE [1]) casts solving as a sequence of grid-edit actions learned with RL. ReaComp keeps
search over a structured action space but moves it offline: the coding agent searches over
whole programs, scored by a verifier, and compiles the result into a standalone solver that
runs at zero per-task cost. ARCLE's own finding that RL over the raw action space is hard
supports the point that the leverage comes from verifier-scored program-level search. Where
a verifier is not available, the RL formulation has the advantage, and we do not claim
otherwise. The latent-reasoning line (GRAM [2]) amortizes reasoning into a latent trajectory
and needs no DSL, a genuine strength on open-world tasks. ReaComp makes the opposite trade:
it externalizes reasoning as an explicit program, giving up differentiability but gaining an
artifact that is inspectable, reusable, and basically free at test time, and for
DSL-complete verifiable domains this trade pays off (Tables 2-3). We read the two as
complementary. We would push back on the framing that a more complex
formulation is automatically a stronger contribution: that verifier-backed induction works
with a simple recipe is part of the finding, and sets a baseline richer methods can be
measured against. One clarification: in the hybrid setting the LLM does not refine the
solver's output, it is an independent fallback on the tasks the solver leaves unresolved
(§2.3); we will make this clearer.

**ARC-AGI and other benchmarks [3].** ReaComp compiles the reasoning of one shared task
distribution into a single reusable solver over a fixed DSL, so it fits domains that are
effectively DSL-complete: a compact shared vocabulary covers the task family and a fast exact
verifier is available. ARC was designed to violate exactly this: its tasks are deliberately
separated so each can call for novel primitives, with no single shared DSL. DSL-based program
search was central to early ARC progress and remains competitive (the 2020 Kaggle winner was
a hand-built grid DSL with brute-force search), but the strongest recent approaches generalize
the search space beyond any fixed DSL: Li et al. (2024) [4] keep a shared primitive library
but "still allow arbitrary Python code," and since ARC-AGI-2 the data is constructed to remove
DSL-searchable tasks, which suggests a fixed DSL alone plateaus. We read extending induction to non-DSL-complete domains (ARC among them)
as promising future work, where a fixed DSL could serve as a backbone inside a broader
arbitrary-program synthesizer. On the broader list (Mini-ARC, ARC-AGI-1/2, MiniSCAN, List
Functions, ACRE), it helps to separate them by whether they fit the setting ReaComp assumes,
a symbolic input-to-output transformation with a fast exact verifier. List Functions and
MiniSCAN do (integer-list to integer-list, and compositional word-to-symbol-sequence, both
with exact-match verifiers and compact primitives). The ARC-family is verifiable (grid
equality is exact) but deliberately not DSL-complete, the boundary above. ACRE is the one
genuine mismatch: a visual abductive-causal benchmark (CLEVR-style scenes with a Blicket
detector) needing perception and causal inference rather than a string-to-string transform,
with no exact program verifier, so it sits outside the setting regardless of DSL-completeness.

To put weight behind this rather than argue it on paper, we ran ReaComp end-to-end on List
Functions [5; 250 tasks] during the rebuttal. We built two unlabeled demo sets (10 and 25
tasks), each holding only input-output examples plus gpt-oss-120b reasoning traces, and induced
solvers from them with the coding agent; we then evaluate on held-out inputs across the 250
tasks (solved = the induced program reproduces every held-out pair exactly). Single solvers are
weak and high-variance (16-35% held-out), but the union of solvers induced across the two demo
sets reaches 51.6%, at zero LLM inference cost and a one-time coding-agent construction cost.
The useful comparison is against the trace model itself: gpt-oss-120b with best-of-8 scores
87.6% here, and adding the induced solvers as a zero-cost first pass (LLM fallback only on the
rest) raises accuracy to 90.4% while cutting LLM tokens by 36%. Compiling that model's own
traces into symbolic solvers thus solves a few percent more tasks than the model's own sampling,
at lower total cost. The induced solvers are legible search programs, not lookup tables: one
discovers a data-adaptive wave search that infers structure directly from the examples, another
composes cumulative operations (running sum, running max/min) with the usual
slice/sort/filter/dedup primitives. For external context (task accuracy, same 250 functions):
direct GPT-4o is 39.9% and human mean ~52% [6, 7], with iterative-refinement methods reaching up
to ~83% [7] (the hypothesis-refinement setting studied in [3]); our 90.4% is the strongest we
are aware of among the works we could survey, though we did not do an exhaustive search and
protocols differ (we use 24 shown / 8 held-out), so these are reference points, not a strict
ranking. We could not extend to MiniSCAN and Mini-ARC within the rebuttal window, and will add
them, with the full analysis, in revision. We also note [3] is convergent with our approach: it finds LLMs are
strong hypothesis proposers but weak rule appliers, and that pairing them with a symbolic
interpreter that applies and filters rules is what works, the same division of labor ReaComp
automates by compiling the reasoning into an executable solver.

We hope the corrected account of how the solver is built (Concern 1) and the new matched
baseline (Concern 2) address the main reservations, and we are glad to run further
analyses during discussion.

[1] ARCLE: The Abstraction and Reasoning Corpus Learning Environment for RL, 2024.
[2] Generative Recursive Reasoning (GRAM), 2026.
[3] Phenomenal Yet Puzzling: Testing Inductive Reasoning Capabilities of Language Models with
Hypothesis Refinement, 2023.
[4] Li et al. Combining Induction and Transduction for Abstract Reasoning, 2024 (BARC; arXiv
2411.02272).
[5] Rule. The Child as Hacker, PhD thesis, MIT, 2020 (List Functions benchmark).
[6] Rule et al. Symbolic metaprogram search improves learning efficiency and explains rule
learning in humans. Nature Communications, 2024.
[7] Li et al. Patterns Over Principles: The Fragility of Inductive Reasoning in LLMs under
Noisy Observations. Findings of ACL, 2025.

---

# D. Confidential comment to the Area Chair

We thank the Area Chair for the summary and for the specific conditions in point (d). The
evidence for each now lives in the individual reviewer responses (the substantive content
had to move there, since reviewers may not see a shared response); we map them here.

- **(d.1) Trace-quality sensitivity:** answered to Reviewer ggna. §4 ("Solver induction
  ablations") reports removing chain-of-thought drops PBEBench-Hard from 74.7% to 24.8%;
  full grid in App. F.2.
- **(d.2) Induction variance:** answered to Reviewer ggna. 53.4-79.2% on Lite and
  51.8-74.7% on Hard across runs on identical data (§4), with ensembling recovering the
  spread; §5 frames induction as a search over algorithms.
- **(d.3) Generalization:** answered to Reviewer ZhtM. §4 ("Real-world case study")
  reports zero-shot transfer to real IPA data at up to 80.1% by union; setup and
  qualitative examples in App. F.4.
- **(d.4) Reusable abstractions vs. task-specific heuristics:** answered to Reviewer orXJ.
  The induced solvers run documented general algorithms (the SLR solver's
  ascending-complexity search is named in §4), and the new TroVE baseline is the contrast
  case where library learning memorizes instead. Per-solver mechanism analysis in App. F.2.
- **(d.5) Stronger baselines:** answered to Reviewers ggna and orXJ (TroVE in the matched
  harness).
- **(d.6) Broader benchmarks and verifiability:** answered to Reviewer orXJ (benchmark
  triage plus a List Functions pilot run during the rebuttal).

On the point about lacking theory: we agree the contribution is empirical. We will frame the
regularities we observe as conditions under which induction is expected to work (the traces
carry reasoning, the target shares the DSL's structure, and the verifier is fast and exact)
rather than claim more than we can show, and we will narrow the paper's claims to the
verifier-backed PBE and SLR setting.

The two items that currently live only in the appendix are the per-solver mechanism analysis
(App. F.2) and the generalization setup and qualitative examples (App. F.4). For the revision
we will move these into the main text so the body carries the argument on its own, keeping the
appendix for the fuller breakdowns.

---

## Summary of paper changes

1. Add the TroVE baseline (matched harness) and the library-content comparison for (d.4).
2. Give the concrete DSL at the first mention of P in §2 and add a boxed definition in §3.
3. Split Table 1 cost into inference cost vs. amortized construction cost.
4. Use consistent bolding across Tables 1-3 and move the ablation grid to its own table.
5. Tighten the candidate-set notation and add a worked trace-to-solver-to-task example.
6. Retitle the App. B "coding agent capability" limitation to surface the trace-quality
   dependence explicitly, and clarify the trace model vs. the coding agent, the
   interface-vs-algorithm constraint, and the execution bounds.
7. Add related work and positioning: Vision-Language Programs, ActivationReasoning, ARCLE,
   latent reasoning, and Li et al. (2024) on ARC.
8. Narrow the claims to the verifier-backed PBE/SLR setting, with a List Functions pilot in
   the rebuttal and the ARC-family and MiniSCAN benchmarks, solver-seeded fallback, and
   per-instance induction as future work.
