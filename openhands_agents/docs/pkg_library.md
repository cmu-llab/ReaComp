# PkgLibrary — shared function store

`openhands_agents/pkg_library.py`

## Storage model

Each function lives as a standalone `.py` file:

```
pkg_dir/
  __init__.py          # auto-generated: `from .fn_name import fn_name`
  fn_name.py           # contains exactly one top-level def fn_name(...)
  _meta.json           # name → {description, usage_count, code}
```

The directory is a valid Python package bound read-only into the Apptainer container at `/exec/<basename(pkg_dir)>`. Generated code imports via `from library import fn` or `from toolbox import fn`.

**Standalone constraint**: functions must NOT import from the same package. This allows trimming individual files without breaking dependents.

## Key methods

| Method | Description |
|--------|-------------|
| `add(name, description, code)` | Write `.py`, update `__init__.py` and `_meta.json`. Thread-safe. |
| `remove(name)` | Delete `.py`, update `__init__.py` and `_meta.json`. Thread-safe. |
| `retrieve(query, k)` | BM25-style: Jaccard on `name + description` tokens. Returns `[{name, description, code}]`. Thread-safe snapshot. |
| `as_listing()` | `name: description` for all functions. |
| `as_listing_with_signatures()` | `def sig(...)  # description` for all functions. Used by TroVE. |
| `increment_usage(names)` | Increment `usage_count` for named functions. Thread-safe. |
| `trim(n_processed, c=0.5)` | Remove functions with usage < c × log10(n_processed). |
| `to_dict()` / `from_dict()` | Checkpoint serialisation. |

## Thread safety

All write operations (`add`, `remove`, `increment_usage`, `trim`) hold `self._lock` (a `threading.Lock`). `retrieve` takes a snapshot of `_meta` under the lock before scoring. This makes parallel task execution safe: multiple threads can call `execute_code` (which calls `increment_usage`) and `add_to_library` concurrently.

## Checkpoint / resume

`_meta.json` is the source of truth. On resume, `_load_meta()` restores the in-memory dict from disk; the `.py` files are already in place. `recompute_embeddings()` is not needed (no sentence-transformers here — BM25 only).

## Retrieval scoring

```
score(fn) = Jaccard(q_tokens, tokenize(name + " " + description))
```

Tokens: `re.findall(r"[a-z0-9]+", text.lower())`. Code is intentionally excluded — it was noise in earlier experiments (see main `library.py` scoring notes).
