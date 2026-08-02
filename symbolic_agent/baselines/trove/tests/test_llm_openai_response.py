"""Unit tests for TroVELLMClient OpenAI/vLLM response extraction."""

from types import SimpleNamespace

from symbolic_agent.baselines.trove.llm import TroVELLMClient


class _FakeCompletions:
    def create(self, **kwargs):
        msg = SimpleNamespace(content="", reasoning="**Solution**\n```python\nprint('ok')\n```")
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2, completion_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def _client_with_fake_openai_response():
    client = object.__new__(TroVELLMClient)
    client.backend = "openai"
    client._client = _FakeClient()
    client._task_log = []
    client._task_tokens = {"input": 0, "output": 0, "reasoning": 0}
    client._session_tokens = {"input": 0, "output": 0, "reasoning": 0}
    client._debug_dir = None
    return client


def test_openai_call_reads_vllm_reasoning_field_when_content_empty():
    client = _client_with_fake_openai_response()

    raw = client._call_openai("prompt", "openai/gpt-oss-20b", 128, "tag")

    assert "print('ok')" in raw
    assert "print('ok')" in client.get_task_log()[0]["response"]["content"]
