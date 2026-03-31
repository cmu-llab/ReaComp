"""ReGAL Function — faithful reimplementation of codebank/function.py.

Stores one helper function with its source code, description, and a
running record of success/failure across programs that used it.

compute_success() implements blame-normalized scoring from §A.2 of the paper:
  score = Σ(+1 for success, -1/n_funcs for failure) / total
where n_funcs is the number of helper functions in the program, so blame for
a failure is spread across all helpers used.
"""

import ast
import re
from typing import List, Optional


class RegalFunction:
    """
    One entry in the ReGAL CodeBank.

    Attributes
    ----------
    name : str
    args : list[str]
    description : str
        Extracted from the first comment/docstring line after `def`.
    code : str
        Full source code (including `def` line and body).
    round_added : int | None
        Training batch index when this function was first added.
    was_success : list[bool]
        One entry per program that used this function — did the program pass?
    num_programs_used : list[int]
        Parallel to was_success — how many helpers did that program use?
    """

    def __init__(
        self,
        name: str,
        args: List[str],
        description: str,
        code: str,
        round_added: Optional[int] = None,
    ):
        self.name = name
        self.args = args
        self.description = description
        self.code = code
        self.round_added = round_added

        self.was_success: List[bool] = []
        self.num_programs_used: List[int] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_str(cls, code_str: str, round_added: Optional[int] = None) -> "RegalFunction":
        """Parse a function definition string into a RegalFunction."""
        code_str = code_str.strip()
        try:
            tree = ast.parse(code_str)
        except SyntaxError as exc:
            raise SyntaxError(f"Could not parse function: {exc}\n{code_str[:200]}")

        if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError(f"Expected a function definition, got: {type(tree.body[0]) if tree.body else 'empty'}")

        func_node = tree.body[0]
        name = func_node.name
        args = [arg.arg for arg in func_node.args.args]

        # Extract description from first comment or docstring
        description = cls._extract_description(code_str, name)

        return cls(name, args, description, code_str, round_added)

    @staticmethod
    def _extract_description(code_str: str, name: str) -> str:
        """Extract description from first comment line or docstring after def."""
        lines = [l.strip() for l in code_str.split("\n")]
        # Skip the def line itself (index 0)
        if len(lines) > 1:
            second = lines[1]
            if second.startswith("#"):
                # comment-style
                return second.lstrip("#").strip()
            if second.startswith('"""') or second.startswith("'''"):
                # docstring-style: collect until closing triple-quote
                q = second[:3]
                if second.count(q) >= 2 and len(second) > 6:
                    # single-line docstring: """text"""
                    return second.strip(q).strip()
                # multi-line: collect
                desc_lines = [second.lstrip(q)]
                for line in lines[2:]:
                    if q in line:
                        break
                    desc_lines.append(line)
                return " ".join(desc_lines).strip()
        return name  # fall back to function name

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_success(self):
        """
        Compute blame-normalized success score.

        Returns
        -------
        (score, n_used) : (float, int)
            score in [-1, 1].  n_used = len(was_success).
        """
        if not self.was_success:
            return -1.0, 0
        total = 0.0
        for success, n_funcs in zip(self.was_success, self.num_programs_used):
            blame = 1.0 / max(n_funcs, 1)
            total += 1.0 if success else (-1.0 * blame)
        return total / len(self.was_success), len(self.was_success)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def summarize(self, include_success: bool = False) -> str:
        """
        Return a compact string for inclusion in prompts.
        Shows the def line, optional success rate comment, and body.
        Faithful to Function.summarize() in the original.
        """
        lines = self.code.split("\n")
        def_line = lines[0]
        rest = "\n".join(lines[1:])
        if include_success:
            score, n = self.compute_success()
            s_str = f"    # Success rate: {sum(self.was_success)}/{len(self.was_success)}"
            return f"{def_line}\n{s_str}\n{rest}\n"
        return f"{def_line}\n{rest}\n"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "args": self.args,
            "description": self.description,
            "code": self.code,
            "round_added": self.round_added,
            "was_success": self.was_success,
            "num_programs_used": self.num_programs_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegalFunction":
        f = cls(d["name"], d["args"], d["description"], d["code"], d.get("round_added"))
        f.was_success = d.get("was_success", [])
        f.num_programs_used = d.get("num_programs_used", [])
        return f

    def __repr__(self) -> str:
        score, n = self.compute_success()
        return f"RegalFunction({self.name!r}, score={score:.2f}, n={n})"
