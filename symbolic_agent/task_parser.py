"""Task Parser.

Converts a natural-language reasoning prompt into a structured TaskSpec
(domain, I/O types, operation hints, example symbolic inputs).
The spec drives domain-aware function retrieval and guides the SSL/BCR agents.
"""

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# Valid task domains.  Used in retrieval affinity scoring.
DOMAINS = [
    "list_manipulation",
    "string_manipulation",
    "sequence",
    "math",
    "logic",
    "grid",
    "symbolic",
    "other",
]

_TOOLS = [
    {
        "name": "parse_task",
        "description": "Extract the structural components of a symbolic reasoning task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "enum": DOMAINS,
                    "description": "Primary domain of the task.",
                },
                "input_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Python type annotations for the function's parameters, "
                        "e.g. ['list[int]', 'int']."
                    ),
                },
                "output_type": {
                    "type": "string",
                    "description": "Python type annotation for the return value, e.g. 'list[int]'.",
                },
                "operation_hints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key operations implied by the task, e.g. ['filter', 'sort'].",
                },
                "symbolic_inputs": {
                    "type": "string",
                    "description": (
                        "A short Python snippet showing what example inputs look like as "
                        "data structures, e.g. 'lst = [1, 2, 3, 4]'."
                    ),
                },
            },
            "required": [
                "domain", "input_types", "output_type",
                "operation_hints", "symbolic_inputs",
            ],
        },
    }
]

_SYSTEM = (
    "You are a task analyzer for a symbolic reasoning system. "
    "Given a task described in natural language, extract its structural components. "
    "Use standard Python type notation (list[int], str, int, tuple, dict, etc.)."
)


@dataclass
class TaskSpec:
    """Structured representation of a reasoning task, parsed from its NL prompt."""
    original_prompt: str
    domain: str = "symbolic"
    input_types: List[str] = field(default_factory=list)
    output_type: str = ""
    operation_hints: List[str] = field(default_factory=list)
    symbolic_inputs: str = ""

    def summary(self) -> str:
        return (
            f"domain={self.domain}  "
            f"inputs={self.input_types}  "
            f"output={self.output_type}  "
            f"hints={self.operation_hints}"
        )


class TaskParser:
    """Lightweight LLM call to turn an NL prompt into a TaskSpec."""

    def __init__(self, client, model: str = "claude-haiku-4-5-20251001"):
        self.client = client
        self.model = model

    def parse(self, prompt: str) -> TaskSpec:
        try:
            response = self.client.create(
                model=self.model,
                max_tokens=512,
                system=_SYSTEM,
                messages=[{"role": "user", "content": f"Parse this task:\n\n{prompt}"}],
                tools=_TOOLS,
                tool_choice={"type": "any"},
                tag="task_parser",
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "parse_task":
                    inp = block.input
                    spec = TaskSpec(
                        original_prompt=prompt,
                        domain=inp.get("domain", "symbolic"),
                        input_types=inp.get("input_types", []),
                        output_type=inp.get("output_type", ""),
                        operation_hints=inp.get("operation_hints", []),
                        symbolic_inputs=inp.get("symbolic_inputs", ""),
                    )
                    logger.info("TaskParser: %s", spec.summary())
                    return spec
        except Exception as exc:
            logger.warning("TaskParser failed (%s), using fallback spec.", exc)

        return TaskSpec(original_prompt=prompt)
