# Create a markdown file with the requested prompt content

content = """# Library Induction & Compositional Generalization Research Prompt

## Overview
You are assisting with a research project on **neurosymbolic tool (library) induction** for improving compositional generalization in programmatic reasoning tasks.

The goal is to learn a **minimal set of reusable Python tools (functions/programs)** under a **budget constraint**, such that these tools:
- Generalize from **easy → hard tasks**
- Enable solving more complex tasks via **composition**
- Approximate an **optimal (minimal) tool library**

---

## Core Research Objective

Learn a library \\( L \\) of tools such that:

- Each tool is a Python function
- The library size is constrained (budget)
- Tools are reusable across tasks
- The induced library improves performance on harder tasks

We evaluate:
- **Solve rate vs difficulty (curriculum tiers)**
- **Generalization gap (easy → hard)**
- **Library size vs performance tradeoff**
- **Solution complexity (description length / minimal program size)**
- **Gap to oracle/minimal solution**

---

## Key Datasets

### 1. PBEBench (Primary)
- Program synthesis / transformation tasks
- Multi-step compositional structure
- Has hardness levels (based on transformation complexity)

### 2. SLR-Bench (Secondary)
- Symbolic logic rule induction
- 20-level curriculum
- Increasing complexity via:
  - rule length
  - predicate complexity
  - problem size

### (Optional) Reasoning Gym
- Used only for transfer/generalization experiments

---

## Key Concepts

### Compositional Generalization
Performance should degrade gracefully as:
- number of steps increases
- rule depth increases
- composition becomes harder

### Solution Complexity (Important)
We explicitly track:
- minimal program / rule length
- description length
- simplicity of learned solution

This acts as an **Occam-style objective**:
> Prefer simpler reusable tools that compose well

---

## Tool Definition

A tool is:
- A Python function
- Encapsulates reusable logic
- Can be composed with other tools
- Can be invoked by an agent or program generator

---

## Tasks for You

You will help **reproduce and adapt baseline methods** into this framework.

### Target Baselines
1. **ReGAL**
2. **TroVE**

These must be adapted to:
- operate on Python tools
- run on PBEBench and SLR-Bench
- use a shared execution + evaluation interface

---

## Implementation Requirements

### 1. Unified Framework
All methods should:
- use the same input/output format
- use Python tool execution
- share evaluation code

---

### 2. Faithful Reproduction
Preserve:
- core algorithmic idea
- tool generation mechanism
- training signal (reward, filtering, etc.)

Adapt:
- input/output formatting
- dataset interface
- execution backend

---

### 3. Minimal Engineering
Avoid:
- complex multi-agent systems
- unnecessary pipelines

Focus on:
- correctness
- clarity
- reproducibility

---

## Expected Components

### Tool Library
- storage of tools (Python functions)
- ability to add/update tools
- ability to call tools during solving

---

### Solver
- uses tools to solve tasks
- may generate new tools
- may compose tools

---

### Evaluation
For each task:
- success/failure
- number of steps
- tools used
- solution complexity

---

## Metrics to Implement

- Accuracy / solve rate
- Performance vs difficulty tier
- Library size
- Tool reuse frequency
- Solution complexity (program length)
- Gap to oracle (if available)

---

## Baseline Adaptation Notes

### ReGAL
- Focus on tool generation + refinement loop
- Adapt tools to Python functions
- Ensure compositional reuse

### TroVE
- Focus on tool discovery and reuse
- Ensure tools are callable and composable
- Avoid retrieval-only behavior

---

## Important Constraints

- Do NOT reduce this to tool retrieval
- Tools must represent **reusable abstractions**
- Composition must be explicit

---

## Sanity Checks

Before full experiments:
- Verify tools are reused across tasks
- Verify performance increases with tools
- Verify harder tasks benefit more

---

## Deliverables

- Clean implementation of ReGAL and TroVE
- Integrated into shared framework
- Runs on PBEBench and SLR-Bench
- Logs all metrics above

---

## Guiding Principle

This is NOT just about performance.

This is about:
> Learning minimal, reusable abstractions that improve compositional generalization.

Keep implementations simple, principled, and aligned with this goal.
"""

file_path = "/mnt/data/library_induction_prompt.md"
with open(file_path, "w") as f:
    f.write(content)

file_path