"""Task Parser.

Converts a natural-language reasoning prompt into a structured TaskSpec
(domain, I/O types, operation hints, example symbolic inputs).
The spec drives domain-aware function retrieval and guides the SSL/BCR agents.

Responds with a plain JSON object — no tool calling.
"""

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

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

_SYSTEM = (
    "You are a task analyzer for a symbolic reasoning system. "
    "Given a task described in natural language, extract its structural components. "
    "Use standard Python type notation (list[int], str, int, tuple, dict, etc.).\n\n"
    "Respond with exactly this JSON:\n"
    "{\n"
    '  "domain": "<one of: list_manipulation, string_manipulation, sequence, math, logic, grid, symbolic, other>",\n'
    '  "input_types": ["<Python type annotation for each parameter>"],\n'
    '  "output_type": "<Python type annotation for the return value>",\n'
    '  "operation_hints": ["<key operations implied by the task>"],\n'
    '  "symbolic_inputs": "<short Python snippet showing what example inputs look like>"\n'
    "}"
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

    def __init__(self, client, model: str = "claude-sonnet-4-5"):
        self.client = client
        self.model = model

    def parse(self, prompt: str) -> TaskSpec:
        try:
            result = self.client.create(
                model=self.model,
                max_tokens=512,
                system=_SYSTEM,
                messages=[{"role": "user", "content": f"Parse this task:\n\n{prompt}"}],
                tag="task_parser",
            )
            if result:
                spec = TaskSpec(
                    original_prompt=prompt,
                    domain=result.get("domain", "symbolic"),
                    input_types=result.get("input_types", []),
                    output_type=result.get("output_type", ""),
                    operation_hints=result.get("operation_hints", []),
                    symbolic_inputs=result.get("symbolic_inputs", ""),
                )
                logger.info("TaskParser: %s", spec.summary())
                return spec
        except Exception as exc:
            logger.warning("TaskParser failed (%s), using fallback spec.", exc)

        return TaskSpec(original_prompt=prompt)
