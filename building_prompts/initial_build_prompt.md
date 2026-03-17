# Agentic Symbolic Reasoning System with Library Learning

## Objective

Build an agentic reasoning system that solves symbolic inductive
reasoning tasks by: 1. Constructing and reusing a shared library of
functions 2. Minimizing complexity via reuse (Occam's razor) 3.
Iteratively decomposing problems using learned abstractions

------------------------------------------------------------------------

## Core Principle

Prefer: - Reusing existing functions - Composing simple functions

Avoid: - Creating new functions unless necessary - Long, monolithic
solutions

------------------------------------------------------------------------

## System Architecture

### State Object

``` python
State = {
    "task_input": None,
    "task_type": "",
    "working_memory": None,
    "library": [],
    "trace": [],
    "budget": 0.0,
    "steps": 0,
    "solved": False,
    "solution": None
}
```

------------------------------------------------------------------------

### Function Representation

``` python
class Function:
    def __init__(self, name, code):
        self.name = name
        self.code = code
        self.embedding = None
        self.usage_count = 0
        self.creation_cost = 0.0
```

------------------------------------------------------------------------

### Controller

``` python
for step in range(MAX_STEPS):
    if solved(state):
        break

    if should_call_ssl(state):
        state = SSL_agent(state)
    else:
        state = BCR_agent(state)
```

------------------------------------------------------------------------

### SSL Agent

-   Retrieve relevant functions
-   Reuse, compose, or create new
-   Prefer reuse over invention

------------------------------------------------------------------------

### BCR Agent

-   Solve using available functions
-   If not possible, decompose problem

------------------------------------------------------------------------

### Reporting Agent

-   Convert solution to required format
-   No new reasoning allowed

------------------------------------------------------------------------

## Cost Function

``` python
TotalCost = α * NumNewFunctions           + β * TotalFunctionLength           + γ * RedundancyPenalty           - δ * ReuseReward
```

### Components

-   NumNewFunctions: number of new functions created\
-   TotalFunctionLength: total lines of code used\
-   RedundancyPenalty: similarity with existing functions\
-   ReuseReward: log-scaled usage frequency

### Objective

``` python
Objective = TaskLoss + λ * TotalCost
```

------------------------------------------------------------------------

## Function Usefulness Score

``` python
usefulness = usage_count / (creation_cost + 1e-6)
```

------------------------------------------------------------------------

## Constraints

-   Must use at least one library function per solution
-   Limit function length
-   Avoid duplicates
-   Penalize unused functions

------------------------------------------------------------------------

## Sub-Agent Prompts

### SSL Agent

-   Retrieve functions
-   Prefer reuse
-   Create new only if necessary

### BCR Agent

-   Solve or decompose
-   Prefer composition

### Controller

-   Decide between SSL and BCR

### Reporting

-   Translate only

------------------------------------------------------------------------

## Deliverables

-   Modular codebase
-   Example runs
-   Logging of function reuse and cost

------------------------------------------------------------------------

## Goal

Demonstrate: - Function reuse across tasks - Reduced complexity over
time - Emergent abstraction