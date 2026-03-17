"""
Example symbolic inductive reasoning tasks.

Each task is a dict with:
  "type"   – task category label
  "input"  – the task description / specification passed to the agents
"""

TASKS = [
    # -----------------------------------------------------------------------
    # List transformation tasks
    # -----------------------------------------------------------------------
    {
        "type": "list_transform",
        "input": {
            "description": "Given a list of integers, return a new list with each element doubled.",
            "examples": [
                {"input": [1, 2, 3], "output": [2, 4, 6]},
                {"input": [0, -1, 5], "output": [0, -2, 10]},
            ],
        },
    },
    {
        "type": "list_transform",
        "input": {
            "description": "Given a list of integers, return only the even numbers.",
            "examples": [
                {"input": [1, 2, 3, 4, 5, 6], "output": [2, 4, 6]},
                {"input": [7, 8, 9, 10], "output": [8, 10]},
            ],
        },
    },
    {
        "type": "list_transform",
        "input": {
            "description": "Reverse a list.",
            "examples": [
                {"input": [1, 2, 3], "output": [3, 2, 1]},
                {"input": ["a", "b", "c"], "output": ["c", "b", "a"]},
            ],
        },
    },
    # -----------------------------------------------------------------------
    # Sequence / pattern tasks
    # -----------------------------------------------------------------------
    {
        "type": "sequence",
        "input": {
            "description": (
                "Given a list, return the sum of all elements."
            ),
            "examples": [
                {"input": [1, 2, 3, 4], "output": 10},
                {"input": [10, -5, 3], "output": 8},
            ],
        },
    },
    {
        "type": "sequence",
        "input": {
            "description": (
                "Given a list of integers, return the running cumulative sum "
                "(each element is the sum of all preceding elements including itself)."
            ),
            "examples": [
                {"input": [1, 2, 3, 4], "output": [1, 3, 6, 10]},
                {"input": [5, 5, 5], "output": [5, 10, 15]},
            ],
        },
    },
    # -----------------------------------------------------------------------
    # String / symbolic tasks
    # -----------------------------------------------------------------------
    {
        "type": "string_transform",
        "input": {
            "description": "Given a string, return it with every word capitalised.",
            "examples": [
                {"input": "hello world", "output": "Hello World"},
                {"input": "symbolic reasoning", "output": "Symbolic Reasoning"},
            ],
        },
    },
    {
        "type": "string_transform",
        "input": {
            "description": (
                "Given a list of strings, return a single string formed by joining "
                "them with a comma and a space."
            ),
            "examples": [
                {"input": ["a", "b", "c"], "output": "a, b, c"},
                {"input": ["hello", "world"], "output": "hello, world"},
            ],
        },
    },
    # -----------------------------------------------------------------------
    # Higher-order / compositional tasks
    # -----------------------------------------------------------------------
    {
        "type": "compositional",
        "input": {
            "description": (
                "Given a list of integers, double each element and then keep only "
                "the even results."
            ),
            "examples": [
                {"input": [1, 2, 3, 4], "output": [2, 4, 6, 8]},
            ],
        },
    },
    {
        "type": "compositional",
        "input": {
            "description": (
                "Given a list of integers, return the sum of the squares of the odd numbers."
            ),
            "examples": [
                {"input": [1, 2, 3, 4, 5], "output": 35},   # 1 + 9 + 25
                {"input": [2, 4, 6], "output": 0},
            ],
        },
    },
]
