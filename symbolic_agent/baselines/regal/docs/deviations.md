# ReGAL Deviations from Paper

This document lists all known deviations from the paper
"Refactoring for Generalizable Abstraction Learning" (Stengel-Eskin et al., ICML 2024).

---

## 1. Chat API instead of completion API

**Paper:** Uses `gpt-3.5-turbo-0613` via the completion API (prompt → continuation).

**Our implementation:** Uses the chat messages API (user turn → assistant response)
for both Anthropic (claude-*) and OpenAI-compatible (vLLM) backends.

**Impact:** Prompts are formatted as user messages rather than raw completions. The
LLM may be less likely to follow the exact formatting constraints (like continuing
from "NEW PROGRAM 1:") since it generates a full response rather than completing a
partial sequence. Kept identical prompt text to the paper to minimize this gap.

---

## 2. Embedding model: sentence_transformers instead of OpenAI Ada-002

**Paper:** Uses OpenAI's `text-embedding-ada-002` for query clustering and
CodeBank/DemoBank retrieval (§A.3).

**Our implementation:** Uses `sentence_transformers` (default: `all-MiniLM-L6-v2`)
for all embeddings. ChromaDB mode also uses sentence_transformers rather than Ada-002.

**Why:** Ada-002 requires an OpenAI API key and incurs per-token cost. Sentence
transformers are local, free, and compatible with any backend.

**Impact:** Slightly different clustering and retrieval quality. The embedding
space may differ, affecting which examples are grouped together during training
and which helpers are retrieved at test time.

---

## 3. ChromaDB API: new PersistentClient instead of deprecated duckdb+parquet

**Paper:** References `chromadb` with the original `duckdb+parquet` backend (§A.3,
footnote 6: https://github.com/chroma-core/chroma/).

**Our implementation:** Uses `chromadb.PersistentClient(path=...)` (new API since
chromadb ≥0.4.0). The old `duckdb+parquet` backend was removed in chromadb 0.4.0.

**Impact:** Functionally equivalent. The persistence format differs but the
retrieval interface is the same.

---

## 4. No comment-adding preprocessing step

**Paper:** Optionally adds comments to primitive programs before refactoring
(§3.1, Tables 11–12). Used for LOGO (Table 9: Add comments = True).

**Our implementation:** Comments are not added. This is an optional preprocessing
step and is domain-specific. The prompts for comment decomposition (Table 11) and
comment addition (Table 12) are not implemented.

**Impact:** May reduce the quality of refactored programs for domains where
query-code alignment is non-obvious. On PBEBench and reasoning_gym string tasks,
the queries are short and self-explanatory, so this is expected to have minimal impact.

---

## 5. Single-epoch training by default

**Paper:** Runs 1 epoch for most domains, 3 epochs for LOGO (Table 9).

**Our implementation:** Default `n_epochs=1`. Can be increased via training code
(not exposed as a CLI flag yet).

**Impact:** Fewer rounds of refactoring may produce a smaller or less refined CodeBank.

---

## 6. solve_with_reward is a one-shot wrapper (no online retry loop)

**Paper:** ReGAL has no online retry loop at test time — the agent generates one
program per test query.

**Our implementation:** `solve_with_reward()` calls `solve()` once and evaluates the
reward. There is no iterative feedback loop (unlike SSL_BCR's `max_reward_iters`).

**Why:** Faithful to the paper — ReGAL is an offline training method.

---

## 7. add_comments skipped (no gold programs for comment generation)

**Paper:** Comments are generated using a zero-shot LLM call per example (§A.1).
This requires a gold primitive program and an LLM call per example before training.

**Our implementation:** Skipped. No comment preprocessing is done before refactoring.

**Why:** This is optional (only used for LOGO in Table 9). Our target domains
(PBEBench, reasoning_gym) have short queries where comments add little value.

---

## 8. Ward's clustering: n_clusters = len(data) // batch_size

**Paper:** Clusters examples into batches using Ward's hierarchical clustering,
forming a tree that is topologically sorted and grouped into batches of size k.

**Our implementation:** Uses `AgglomerativeClustering(n_clusters=n_batches, linkage="ward")`
where `n_batches = len(data) // batch_size`. This approximates the paper's batch
formation without implementing the full topological tree sort. The curriculum sort
(by avg query length) is preserved.

**Impact:** Batch boundaries may differ slightly from the paper's tree traversal order.
Functionally equivalent for most practical purposes.

---

## 9. Training data format

**Paper:** Uses domain-specific training sets with gold primitive programs.

**Our implementation:** Uses our task JSONL format (same as SSL_BCR input) with an
additional `program` key in each record's `entry` dict containing the primitive program.
The `get_question()` function extracts the query from `question`/`prompt`/`task` keys.

**Impact:** Requires users to add `"program"` keys to their training JSONL files.
No change to the refactoring logic.

---

## Summary Table

| # | Aspect | Paper | Our Implementation |
|---|---|---|---|
| 1 | LLM API | completion (gpt-3.5-turbo-0613) | chat messages (Anthropic/OpenAI-compat) |
| 2 | Embeddings | OpenAI Ada-002 | sentence_transformers (local) |
| 3 | ChromaDB API | duckdb+parquet (deprecated) | PersistentClient (chromadb ≥0.4) |
| 4 | Comment preprocessing | optional (LOGO: True) | not implemented |
| 5 | Training epochs | 1–3 (domain-specific) | 1 (default) |
| 6 | Test-time retry | n/a (one-shot) | one-shot (faithful) |
| 7 | add_comments | GPT-3.5 per example | skipped |
| 8 | Clustering | topological tree sort | AgglomerativeClustering + curriculum sort |
| 9 | Training data | domain-specific gold programs | task JSONL + `program` key |
