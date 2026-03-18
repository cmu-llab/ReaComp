# Function Library and Retrieval

**File:** `symbolic_agent/library.py`

The `FunctionLibrary` is the central shared state across all tasks in a session. Every agent reads from it; SSL writes to it.

---

## Core Operations

```python
library = FunctionLibrary()

# Add or update a function (upsert by name)
library.add(func: Function)

# Look up by exact name
func = library.get("filter_even")   # returns None if not found

# Remove a function
library.remove("filter_even")       # returns True if it existed

# Retrieve top-k relevant functions for a task
candidates = library.retrieve_relevant(query="filter even numbers", task_spec=spec, top_k=5)

# Format the library for inclusion in a prompt
prompt_block = library.format_for_prompt()

len(library)   # number of functions
```

---

## Retrieval Scoring

`retrieve_relevant()` scores every function in the library against the query using four signals:

```
score = 0.5 × text_jaccard
      + 0.3 × domain_affinity
      + 0.1 × type_overlap
      + 0.1 × log_usage
```

### Text Jaccard (0.5 weight)

Tokenises both the query and `f"{func.name} {func.description} {func.code}"` into word-level tokens, then computes `|intersection| / |union|`. Handles exact keyword matches well; no embedding needed.

### Domain Affinity (0.3 weight)

Uses the `DOMAIN_AFFINITY` matrix (see below) to score how well a function from domain A transfers to a task in domain B. Requires `task_spec` to be provided.

### Type Overlap (0.1 weight)

Compares base Python types (e.g. `list[int]` → `list`, `str` → `str`). Jaccard over base-type sets of the task's `input_types` vs the function's `input_types`. Forgiving about generic parameters while still distinguishing `list` from `str` from `int`. Requires `task_spec`.

### Log Usage (0.1 weight)

`log(1 + usage_count)`. Gives a small boost to functions that have proven useful before without letting popular functions dominate.

---

## Domain Affinity Matrix

Cross-domain transfer scores. Key values:

| Task domain → | list_manip | string | sequence | math | logic | grid |
|---|---|---|---|---|---|---|
| **list_manip** | 1.0 | 0.4 | 0.6 | 0.1 | 0.1 | 0.2 |
| **string** | 0.4 | 1.0 | 0.3 | 0.0 | 0.1 | 0.1 |
| **sequence** | 0.6 | 0.3 | 1.0 | 0.5 | 0.3 | 0.2 |
| **math** | 0.1 | 0.0 | 0.5 | 1.0 | 0.6 | 0.2 |
| **logic** | 0.1 | 0.1 | 0.3 | 0.6 | 1.0 | 0.3 |
| **grid** | 0.2 | 0.1 | 0.2 | 0.2 | 0.3 | 1.0 |

**Design rationale:**
- `list_manipulation` ↔ `sequence`: high (0.6) — list iteration primitives transfer naturally
- `list_manipulation` ↔ `string_manipulation`: medium (0.4) — both process sequences, different element types
- `math` ↔ `logic`: high (0.6) — boolean and numeric reasoning often share structure
- `math` ↔ `string_manipulation`: zero (0.0) — no meaningful transfer
- `general` domain: score 0.5 against everything — mild universal utility

Functions tagged `domain="general"` are given a flat 0.5 affinity against all task domains.

---

## Prompt Formatting

`format_for_prompt()` renders the full library as a markdown block inserted into SSL and BCR prompts:

```markdown
# Available Library Functions

## filter_even  [list_manipulation]
# Return only even integers from a list  inputs: list[int]  →  list[int]
```python
def filter_even(lst: list[int]) -> list[int]:
    return [x for x in lst if x % 2 == 0]
```

## double_elements  [list_manipulation]
...
```

If the library is empty, returns `"# Library is empty"`.
