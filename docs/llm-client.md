# LLM Client

**File:** `symbolic_agent/llm_client.py`

`LLMClient` is a thin backend-agnostic adapter. All four agents call `client.create()` and receive a plain Python `dict`. The client handles backend differences, JSON parsing, and debug logging invisibly.

---

## Initialisation

```python
# Anthropic
client = LLMClient(backend="anthropic", api_key="sk-ant-...")

# vLLM or any OpenAI-compatible endpoint
client = LLMClient(backend="openai", base_url="http://localhost:8000/v1")

# With debug logging
client = LLMClient(..., debug_dir="debug_logs/")
```

**Parameters:**

| Param | Type | Description |
|---|---|---|
| `backend` | `"anthropic"` \| `"openai"` | Which SDK to use |
| `base_url` | str | Required for `"openai"` backend |
| `api_key` | str | Optional; falls back to env vars `ANTHROPIC_API_KEY` / `VLLM_API_KEY` |
| `debug_dir` | str | If set, writes one JSON file per call to `<debug_dir>/run_<timestamp>/` |

---

## `create()` — the only public method

```python
result: dict = client.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system="You are ...\n\nRespond with a JSON object...",
    messages=[{"role": "user", "content": "..."}],
    tag="ssl",          # optional — used in debug log filenames
)
```

Returns a `dict` (the parsed JSON response). Returns `{}` on parse failure (always logged as a warning).

The client automatically appends the following to every `system` prompt before sending:

```
Respond with a single valid JSON object and nothing else.
No markdown fences, no prose before or after the JSON.
```

---

## Backends

### Anthropic

Uses the `anthropic` SDK. Sends `system` + `messages` without any tools. Reads the first `text`-type block from `response.content`.

### OpenAI / vLLM

Uses the `openai` SDK. Sends `response_format={"type": "json_object"}` which instructs the server to guarantee valid JSON in `content`. Reads `msg.content`.

The `reasoning_content` field (populated by reasoning models like `gpt-oss-120b`) is captured in debug logs but not used for parsing — the final committed answer is always in `content`.

---

## JSON Parsing (`_parse_json`)

After retrieving raw text from the model:

1. Strip leading/trailing whitespace
2. Strip markdown code fences (` ```json ... ``` `) if present — some models wrap output regardless of instructions
3. `json.loads()` the result
4. If the result is a single-element list containing a dict (e.g. `[{"action": "solve", ...}]`), unwrap it automatically
5. On `JSONDecodeError` or non-dict result: log a warning and return `{}`

---

## Debug Logging

When `debug_dir` is set, a timestamped subdirectory is created at initialisation:

```
<debug_dir>/run_20260318T060621/
  0001_task_parser_20260318T060635_517252.json
  0002_ssl_20260318T060643_455080.json
  0003_bcr_20260318T060652_379541.json
  0004_reporting_20260318T060658_119875.json
  ...
```

Files are written immediately after each call so they survive mid-run crashes.

### Debug log schema

```json
{
  "call_index": 2,
  "timestamp": "20260318T060643_455080",
  "tag": "ssl",
  "model": "openai/gpt-oss-120b",
  "request": {
    "system": "You are the SSL agent ...",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 1024
  },
  "response": {
    "finish_reason": "stop",
    "content": "{\"action\": \"create\", ...}",
    "reasoning_content": "We need to create a function that ..."
  },
  "parsed_result": {
    "action": "create",
    "name": "double_elements",
    "code": "def double_elements(lst: list[int]) -> list[int]:\n    return [x * 2 for x in lst]",
    "domain": "list_manipulation",
    "input_types": ["list[int]"],
    "output_type": "list[int]"
  }
}
```

The `reasoning_content` field shows the model's internal chain-of-thought (GptOss reasoning models only). This is the primary field to inspect when diagnosing unexpected agent behaviour.

---

## Adding a New Backend

Implement two things in `LLMClient`:

1. Add the backend to `__init__` to initialise the SDK client
2. Add a `_create_<backend>()` method that:
   - Sends `system` + `messages` to the API
   - Returns the raw response text
   - Calls `self._parse_json(text, tag)` and `self._write_debug_log(...)` before returning
