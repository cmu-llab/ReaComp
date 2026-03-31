# How ReGAL Works

ReGAL (Refactoring for Generalizable Abstraction Learning) is a gradient-free
method for learning a shared library of helper functions from a small set of
(query, program) examples. It consists of two phases: **training** (offline) and
**testing** (inference).

---

## Training (Algorithm 1)

### 0. Preprocessing

Before training, examples are clustered into related batches and sorted by difficulty.

**Clustering.** Each query is embedded using a sentence transformer model (or OpenAI
Ada-002 in the original paper). The embeddings are hierarchically clustered using
Ward's algorithm (`sklearn.cluster.AgglomerativeClustering(linkage="ward")`), grouping
semantically related examples together so the refactoring LLM sees multiple related
programs at once.

**Curriculum.** Batches are sorted by average query length (shortest → longest).
Shorter queries tend to be simpler programs, whose abstractions can later be reused
in more complex ones.

### Stage 1: refactorBatch

The LLM receives a batch of `k` (query, program) pairs along with the current
CodeBank contents, and is asked to:
1. Write shared helper function definitions (`NEW HELPERS:`).
2. Rewrite each program to use those helpers (`NEW PROGRAM {i}:`).

The LLM produces thought comments (`# Thoughts: ...`) before each program.

### Stage 2a: verify

Each refactored program is executed (with helper code prepended) in a subprocess.
Its stdout is compared to the stdout of the original primitive program:
```
passes = (stdout(refactored + helpers) == stdout(original))
```
Only helper functions that appear in a *passing* refactored program are added to the
CodeBank. The (query, refactored program, success) triple is always added to the
DemoBank regardless of pass/fail.

### Stage 2b: retry

For programs that fail verification, a follow-up prompt is built that includes:
- The original and refactored programs.
- The execution error or stdout mismatch.
- The current helper functions.

The LLM generates corrected versions; these are re-verified. Successful corrections
are added to the CodeBank.

### Stage 3a: editCodeBank (optional, `--regal-edit-codebank`)

Every `edit_every` batches, the controller prompts the LLM to improve each CodeBank
function that has both passing and failing programs. The LLM sees:
- The current function.
- The pass/fail rate.
- One passing and one failing demo (query + program).
- The rest of the CodeBank.

If the edited function passes more demos than the original, it replaces the original.

### Stage 3b: pruneCodeBank

Every `prune_every` batches (and once after all epochs), functions are pruned if their
blame-normalized score ≤ threshold θ=0.0 (default) and they have been used ≥ 3 times.

Score for function f:
```
score = Σ_{p ∈ passing} +1  +  Σ_{p ∈ failing} −1/n_p
```
where `n_p` = number of helpers used in program `p` (blame is shared equally).

---

## Testing (Algorithm 2)

For each test query:

1. **Retrieve helpers.** Up to 20 helper functions are retrieved from the CodeBank
   by cosine similarity between the query and `name: description` strings.

2. **Retrieve ICL examples.** The ICL budget (`--regal-icl-budget`, default 10) is
   split by ratio `--regal-icl-split` (default 0.5):
   - `M_demo = r * M` examples from the DemoBank (success_only=True).
   - `M_train = M - M_demo` primitive training examples.
   Both sets are retrieved by query similarity.

3. **Build agent prompt.** The prompt (Table 14 style) includes:
   - Task instruction.
   - Retrieved helper definitions.
   - ReAct-style thought instruction ("Begin with a # Thought: comment").
   - ICL examples.
   - Test query.

4. **Generate program.** The LLM produces a program (possibly using helpers).

5. **Execute.** The program is run with helper code prepended. Stdout is the answer.

---

## Data Flow

```
Training data:
  (query, primitive_program) pairs
       │
       ▼
  _cluster_and_sort()           Ward's clustering + curriculum sort
       │
       ▼
  refactorBatch()               LLM: rewrite N programs using shared helpers
       │
       ▼
  verify()                      subprocess stdout comparison
       │
       ├─ pass ──────────────▶  CodeBank + DemoBank (success=True)
       │
       └─ fail ──▶ retry() ──▶  CodeBank + DemoBank (success=True/False)
       │
       ▼
  editCodeBank() [optional]     LLM: improve failing helpers
  pruneCodeBank()               Remove helpers with score ≤ θ
```

---

## Key Structures

| Structure | Description |
|---|---|
| `ReGALCodeBank` | Library of verified helper functions. Supports sentence_transformers or chromadb retrieval. |
| `ReGALDemoBank` | Verified (query, program, success) tuples for ICL. sentence_transformers retrieval. |
| `RegalFunction` | One helper: name, args, description, code, was_success[], num_programs_used[]. |
| `ReGALController` | Orchestrates training + test-time inference. |

---

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--framework regal` | — | Enable ReGAL |
| `--regal-train-file FILE` | — | JSONL with `program` key for offline training |
| `--regal-retrieval {sentence_transformers,chromadb}` | sentence_transformers | Vector retrieval backend |
| `--regal-embedding-model MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `--regal-codebank-dir DIR` | — | Save/load trained CodeBank+DemoBank |
| `--regal-batch-size N` | 4 | Examples per refactoring batch |
| `--regal-edit-codebank` | off | Enable Stage 3a editCodeBank |
| `--regal-edit-every N` | 5 | Run editCodeBank every N batches |
| `--regal-prune-every N` | 5 | Run pruneCodeBank every N batches |
| `--regal-icl-budget N` | 10 | Total ICL examples in agent prompt |
| `--regal-icl-split R` | 0.5 | Fraction of ICL budget from DemoBank |
