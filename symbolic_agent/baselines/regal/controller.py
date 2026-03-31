"""ReGAL Controller — faithful reimplementation of training + test-time inference.

Training (Algorithm 1 in paper):
    1. Preprocess: embed queries, cluster via Ward's, sort batches by avg query length (curriculum).
    2. For each batch:
       a. refactorBatch: prompt LLM with (query, program) pairs + current codebank.
       b. Parse response → programs + helpers.
       c. verify: run each refactored program, compare stdout to original.
       d. retry: one follow-up call for failed programs with execution error feedback.
       e. Add verified helpers to CodeBank; add all programs to DemoBank.
       f. Every edit_every batches: editCodeBank (if --regal-edit-codebank).
       g. Every prune_every batches: pruneCodeBank.
    3. Optionally prune before testing.

Testing (Algorithm 2):
    1. retrieve up to 20 helpers from CodeBank by query similarity.
    2. retrieve Mdemo=icl_split*icl_budget demos from DemoBank + Mtrain primitive examples.
    3. Build agent prompt (Table 14 style, ReAct thoughts).
    4. Generate program.
    5. Execute program with helper code prepended.
    6. Return stdout as answer.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

from .codebank import ReGALCodeBank, ReGALDemoBank
from .executor import get_func_names, run_program, split_helpers_and_calls, verify
from .function import RegalFunction
from .llm import RegalLLMClient
from .prompts import (
    build_agent_prompt,
    build_edit_codebank_prompt,
    build_refactor_batch_prompt,
    build_retry_prompt,
    get_question,
    parse_edit_result,
    parse_result,
)

logger = logging.getLogger(__name__)


class ReGALController:
    """
    ReGAL: Refactoring for Generalizable Abstraction Learning.

    Parameters
    ----------
    api_key : str
    model : str
    base_url : str | None
        OpenAI-compatible base URL (e.g. for vLLM). If None, uses Anthropic.
    debug_dir : str | None
    retrieval : {'sentence_transformers', 'chromadb'}
        Vector retrieval backend for CodeBank and DemoBank.
    embedding_model : str
        Sentence transformer model for embeddings (training clustering + retrieval).
    chroma_path : str | None
        Persist directory for chromadb (used when retrieval='chromadb').
    edit_codebank : bool
        Whether to run editCodeBank every edit_every batches (default: off).
    edit_every : int
        Run editCodeBank every N batches (paper default: 5).
    prune_every : int
        Run pruneCodeBank every N batches (paper default: 5).
    prune_threshold : float
        Blame-normalized score threshold for pruning (paper default: 0.0).
    icl_budget : int
        Total ICL examples in agent prompt (paper: 10).
    icl_split : float
        Fraction of ICL budget allocated to DemoBank demos vs primitive training examples
        (paper: 0.5 for most domains).
    temperature : float
        LLM temperature for all calls (paper: GPT-3.5 default; we use 0.7 to allow variety
        in refactoring).
    max_tokens : int
        Max tokens per LLM call.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        debug_dir: Optional[str] = None,
        retrieval: str = "sentence_transformers",
        embedding_model: str = "all-MiniLM-L6-v2",
        chroma_path: Optional[str] = None,
        edit_codebank: bool = False,
        edit_every: int = 5,
        prune_every: int = 5,
        prune_threshold: float = 0.0,
        icl_budget: int = 10,
        icl_split: float = 0.5,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.retrieval = retrieval
        self.embedding_model = embedding_model
        self.edit_codebank_enabled = edit_codebank
        self.edit_every = edit_every
        self.prune_every = prune_every
        self.prune_threshold = prune_threshold
        self.icl_budget = icl_budget
        self.icl_split = icl_split

        self.codebank = ReGALCodeBank(
            retrieval=retrieval,
            embedding_model=embedding_model,
            chroma_path=chroma_path,
        )
        self.demobank = ReGALDemoBank(embedding_model=embedding_model)

        # Keep a copy of raw training data for ICL primitive examples
        self._train_data: List[dict] = []

        self.llm = RegalLLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            debug_dir=debug_dir,
        )

    # ------------------------------------------------------------------
    # Training: Algorithm 1
    # ------------------------------------------------------------------

    def train(
        self,
        data: List[dict],
        batch_size: int = 4,
        n_retries: int = 1,
        n_epochs: int = 1,
        prune_before_test: bool = True,
    ) -> None:
        """
        Offline training loop: cluster, refactor, verify, edit, prune.

        Parameters
        ----------
        data : list of task dicts from _load_tasks_file.
            Each item must have: item["entry"]["program"] (gold/primitive program)
            and item["input"] (dict with question/prompt/task keys for query extraction).
        batch_size : int
            Number of examples per refactoring batch (paper: 3–5).
        n_retries : int
            Number of retry passes per failed program (paper: 1).
        n_epochs : int
            Number of passes over the training data (paper: 1–3 for LOGO, 1 for others).
        prune_before_test : bool
            Prune the CodeBank after training completes (recommended: True).
        """
        # Keep training data for primitive ICL examples at test time
        self._train_data = data

        valid_data = [
            d for d in data
            if d.get("entry", {}).get("program")
        ]
        if len(valid_data) < len(data):
            logger.warning(
                "train(): %d/%d items missing 'program' key — skipped.",
                len(data) - len(valid_data), len(data),
            )
        if not valid_data:
            logger.error("train(): no valid training items (all missing 'program' key). Aborting.")
            return

        for epoch in range(n_epochs):
            logger.info("ReGAL training epoch %d/%d (%d examples)", epoch + 1, n_epochs, len(valid_data))
            batches = self._cluster_and_sort(valid_data, batch_size)
            logger.info("Formed %d batches (batch_size=%d)", len(batches), batch_size)

            for batch_idx, batch in enumerate(batches):
                logger.info("Batch %d/%d (%d examples)", batch_idx + 1, len(batches), len(batch))
                self._process_batch(batch, batch_idx, n_retries)

                if self.edit_codebank_enabled and (batch_idx + 1) % self.edit_every == 0:
                    self._run_edit_codebank()

                if (batch_idx + 1) % self.prune_every == 0:
                    pruned = self.codebank.prune(threshold=self.prune_threshold)
                    if pruned:
                        logger.info("Pruned %d functions: %s", len(pruned), pruned)

        if prune_before_test:
            pruned = self.codebank.prune(threshold=self.prune_threshold)
            if pruned:
                logger.info("Pre-test prune removed %d functions: %s", len(pruned), pruned)

        logger.info(
            "Training complete. CodeBank: %d functions. DemoBank: %d demos.",
            len(self.codebank), len(self.demobank),
        )

    def _process_batch(self, batch: List[dict], batch_idx: int, n_retries: int) -> None:
        """Refactor one batch: call → parse → verify → retry → update banks."""
        queries = [get_question(item["input"]) for item in batch]
        programs = [item["entry"]["program"] for item in batch]

        # Stage 1: refactorBatch
        codebank_str = self.codebank.as_str()
        prompt = build_refactor_batch_prompt(
            batch=list(zip(queries, programs)),
            codebank_str=codebank_str,
        )
        response = self.llm.call(prompt, tag=f"refactor_batch_{batch_idx}")

        if not response:
            logger.warning("Batch %d: empty LLM response, skipping.", batch_idx)
            return

        new_programs, helpers_code = parse_result(response, n_programs=len(batch))

        # Run original programs to get reference outputs
        original_outputs = []
        for prog in programs:
            ok, out, err = run_program(prog)
            original_outputs.append(out if ok else "")

        # Stage 2a: verify each refactored program
        failed_items = []
        new_helpers_str = helpers_code.strip()

        for i, (query, orig_prog, new_prog, orig_out) in enumerate(
            zip(queries, programs, new_programs, original_outputs)
        ):
            if not new_prog.strip():
                logger.debug("Batch %d item %d: empty refactored program.", batch_idx, i)
                self.demobank.add(query, orig_prog, "", False)
                continue

            passes, actual_out = verify(
                refactored_code=new_prog,
                original_output=orig_out,
                codebank_code=new_helpers_str,
            )

            if passes:
                self._update_banks_success(
                    query=query,
                    new_program=new_prog,
                    helpers_code=new_helpers_str,
                    helpers_round=batch_idx,
                )
                self.demobank.add(query, new_prog, new_helpers_str, True)
            else:
                error_msg = f"expected: {orig_out!r:.80} | got: {actual_out!r:.80}"
                failed_items.append((query, orig_prog, new_prog, new_helpers_str, error_msg))
                self.demobank.add(query, new_prog, new_helpers_str, False)

        # Stage 2b: retry failed programs
        for _ in range(n_retries):
            if not failed_items:
                break
            failed_items = self._retry_failed(failed_items, batch_idx, original_outputs, queries)

    def _retry_failed(
        self,
        failed_items: List[Tuple],
        batch_idx: int,
        original_outputs: List[str],
        all_queries: List[str],
    ) -> List[Tuple]:
        """One retry pass for failed programs. Returns still-failing items."""
        codebank_str = self.codebank.as_str()
        retry_prompt = build_retry_prompt(
            failed_items=failed_items,
            codebank_str=codebank_str,
        )
        retry_response = self.llm.call(retry_prompt, tag=f"retry_{batch_idx}")
        if not retry_response:
            return failed_items

        n_failed = len(failed_items)
        retry_programs, retry_helpers = parse_result(retry_response, n_programs=n_failed)
        retry_helpers_str = retry_helpers.strip() or failed_items[0][3]  # keep old helpers if none new

        still_failing = []
        for j, (query, orig_prog, _, _, _) in enumerate(failed_items):
            new_prog = retry_programs[j] if j < len(retry_programs) else ""
            if not new_prog.strip():
                still_failing.append(failed_items[j])
                continue

            # Get original output for this query
            orig_out = ""
            for i, q in enumerate(all_queries):
                if q == query:
                    orig_out = original_outputs[i] if i < len(original_outputs) else ""
                    break

            passes, actual_out = verify(
                refactored_code=new_prog,
                original_output=orig_out,
                codebank_code=retry_helpers_str,
            )

            if passes:
                self._update_banks_success(
                    query=query,
                    new_program=new_prog,
                    helpers_code=retry_helpers_str,
                    helpers_round=batch_idx,
                )
                # Update demobank entry to success
                self.demobank.add(query, new_prog, retry_helpers_str, True)
            else:
                error_msg = f"retry failed: expected: {orig_out!r:.80} | got: {actual_out!r:.80}"
                still_failing.append((query, orig_prog, new_prog, retry_helpers_str, error_msg))

        return still_failing

    def _update_banks_success(
        self,
        query: str,
        new_program: str,
        helpers_code: str,
        helpers_round: int,
    ) -> None:
        """
        Add verified helpers to CodeBank and record success attribution.

        For each helper in helpers_code:
          - If new: add to CodeBank with round_added.
          - Record success for all helpers used in new_program.
        """
        if not helpers_code.strip():
            return

        # Parse individual helper functions
        helper_defs, _ = split_helpers_and_calls(helpers_code)
        if not helper_defs.strip():
            return

        # Parse into individual function defs
        try:
            import ast
            tree = ast.parse(helper_defs)
        except SyntaxError:
            logger.debug("Could not parse helpers for bank update.")
            return

        new_func_names = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_code = ast.unparse(node)
                try:
                    func = RegalFunction.from_str(func_code, round_added=helpers_round)
                    if func.name not in self.codebank:
                        self.codebank.add(func)
                    new_func_names.append(func.name)
                except (SyntaxError, ValueError) as exc:
                    logger.debug("Could not add helper %s: %s", ast.unparse(node)[:40], exc)

        # Attribution: record success for all helpers called in new_program
        called = get_func_names(new_program)
        n_helpers_used = max(len([n for n in called if n in self.codebank]), 1)
        for name in called:
            func = self.codebank.get(name)
            if func is not None:
                func.was_success.append(True)
                func.num_programs_used.append(n_helpers_used)

    def _update_banks_failure(self, program: str) -> None:
        """Record failure attribution for helpers called in a failed program."""
        called = get_func_names(program)
        n_helpers_used = max(len([n for n in called if n in self.codebank]), 1)
        for name in called:
            func = self.codebank.get(name)
            if func is not None:
                func.was_success.append(False)
                func.num_programs_used.append(n_helpers_used)

    # ------------------------------------------------------------------
    # Stage 3a: editCodeBank
    # ------------------------------------------------------------------

    def _run_edit_codebank(self) -> None:
        """
        Attempt to improve each CodeBank function by prompting the LLM
        with passing/failing demo cases.
        """
        if not self.demobank.demos:
            return
        for func in list(self.codebank.functions):
            self._edit_one_function(func)

    def _edit_one_function(self, func: RegalFunction) -> None:
        """Edit one function: prompt LLM, verify new version, replace if better."""
        if not func.was_success:
            return

        n_pass = sum(func.was_success)
        n_fail = len(func.was_success) - n_pass
        if n_fail == 0:
            return  # already perfect

        pass_perc = n_pass / len(func.was_success)
        fail_perc = n_fail / len(func.was_success)

        # Find demo programs that used this function
        passing_demos = [
            (d["query"], d["program"])
            for d in self.demobank.demos
            if d["success"] and func.name in get_func_names(d["program"])
        ][:3]
        failing_demos = [
            (d["query"], d["program"])
            for d in self.demobank.demos
            if not d["success"] and func.name in get_func_names(d["program"])
        ][:3]

        if not failing_demos:
            return

        # Other codebank functions (excluding the one being edited)
        other_codebank = "\n\n".join(
            f.summarize() for f in self.codebank.functions if f.name != func.name
        )

        prompt = build_edit_codebank_prompt(
            func_str=func.code,
            func_name=func.name,
            pass_perc=pass_perc,
            fail_perc=fail_perc,
            passing_demos=passing_demos,
            failing_demos=failing_demos,
            codebank_str=other_codebank,
        )

        response = self.llm.call(prompt, tag=f"edit_{func.name}")
        if not response:
            return

        new_code = parse_edit_result(response)
        if not new_code.strip():
            return

        # Verify that the new version still passes existing demos
        new_passes = 0
        old_passes = n_pass

        combined_helpers = new_code + "\n\n" + other_codebank
        for demo in passing_demos:
            q, prog = demo
            _, orig_out = run_program(prog)
            ok, out, _ = run_program(prog, extra_code=combined_helpers)
            if ok and out == orig_out:
                new_passes += 1

        if new_passes > old_passes:
            try:
                new_func = RegalFunction.from_str(new_code, round_added=func.round_added)
                new_func.was_success = func.was_success[:]
                new_func.num_programs_used = func.num_programs_used[:]
                self.codebank.add(new_func)
                logger.info(
                    "editCodeBank: replaced %s (pass %.0f%% → %.0f%%)",
                    func.name, pass_perc * 100, new_passes / max(len(passing_demos), 1) * 100,
                )
            except (SyntaxError, ValueError) as exc:
                logger.debug("editCodeBank: could not parse edited %s: %s", func.name, exc)

    # ------------------------------------------------------------------
    # Clustering + curriculum preprocessing
    # ------------------------------------------------------------------

    def _cluster_and_sort(
        self, data: List[dict], batch_size: int
    ) -> List[List[dict]]:
        """
        Cluster examples by query embedding (Ward's), sort batches by avg query
        length (curriculum: easiest first).

        Falls back to sequential grouping if sentence_transformers is not installed.
        """
        queries = [get_question(item["input"]) for item in data]
        n = len(data)

        if n <= batch_size:
            return [data]

        n_batches = max(1, n // batch_size)

        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.cluster import AgglomerativeClustering
            import numpy as np

            encoder = SentenceTransformer(self.embedding_model)
            embs = encoder.encode(queries, convert_to_numpy=True)

            clustering = AgglomerativeClustering(
                n_clusters=n_batches,
                linkage="ward",
            )
            labels = clustering.fit_predict(embs)

            batches_map: Dict[int, List[int]] = {}
            for i, label in enumerate(labels):
                batches_map.setdefault(int(label), []).append(i)

            batch_indices = list(batches_map.values())

        except ImportError:
            logger.warning(
                "sentence_transformers or sklearn not installed — "
                "falling back to sequential batching (no clustering)."
            )
            # Sequential grouping (no clustering)
            batch_indices = [
                list(range(i, min(i + batch_size, n)))
                for i in range(0, n, batch_size)
            ]

        # Sort batches by average query length (curriculum: shortest → longest)
        batch_indices.sort(
            key=lambda idxs: sum(len(queries[i]) for i in idxs) / max(len(idxs), 1)
        )

        return [[data[i] for i in idxs] for idxs in batch_indices]

    # ------------------------------------------------------------------
    # Test-time inference: Algorithm 2
    # ------------------------------------------------------------------

    def solve(self, task_input: dict, task_type: str = "symbolic") -> dict:
        """
        Generate a program for one test query using CodeBank + DemoBank.

        Returns a result dict compatible with main.py's _append_task_output.
        """
        self.llm.reset_task_log()
        query = get_question(task_input)
        original_prompt = task_input.get("prompt", query)

        # Retrieve helpers (up to 20, paper §A.3)
        helpers = self.codebank.retrieve(query, k=20)
        helpers_str = "\n\n".join(h.summarize() for h in helpers) if helpers else ""

        # Retrieve ICL examples: split between demo bank and primitive training data
        m_demo = int(round(self.icl_split * self.icl_budget))
        m_train = self.icl_budget - m_demo

        demo_examples = self.demobank.retrieve(query, k=m_demo, success_only=True)
        icl = [(d["query"], d["program"]) for d in demo_examples]

        # Fill remaining slots with primitive training examples
        if m_train > 0 and self._train_data:
            train_examples = self._retrieve_primitive_examples(query, m_train)
            icl.extend(train_examples)

        # Build and call agent prompt
        prompt = build_agent_prompt(
            query=query,
            codebank_str=helpers_str,
            icl_examples=icl,
            include_thoughts=True,
        )

        response = self.llm.call(prompt, tag="agent")

        # Extract program from response
        program = self._extract_program(response)

        # Execute the program with helper code prepended
        ok, stdout, stderr = run_program(program, extra_code=helpers_str)

        solved = ok
        answer = stdout if ok else ""

        return {
            "task_type": task_type,
            "original_prompt": original_prompt,
            "solved": solved,
            "steps": 1,
            "trace": [
                {
                    "step": 0,
                    "agent": "regal_agent",
                    "action": "generate",
                    "program": program,
                    "exec_ok": ok,
                    "stdout": stdout,
                }
            ],
            "solution": program,
            "final_output": {
                "answer": answer,
                "explanation": "",
                "confidence": 1.0 if ok else 0.0,
                "execution_result": stdout,
                "error": stderr if not ok else None,
            },
            "library_snapshot": [
                {"name": f.name, "description": f.description}
                for f in self.codebank.functions[:20]
            ],
            "cost_summary": {"framework": "regal", "codebank_size": len(self.codebank)},
            "agent_messages": self.llm.get_task_log(),
        }

    def _retrieve_primitive_examples(
        self, query: str, k: int
    ) -> List[Tuple[str, str]]:
        """Retrieve k primitive (non-refactored) training examples by query similarity."""
        valid = [
            d for d in self._train_data
            if d.get("entry", {}).get("program")
        ]
        if not valid:
            return []
        k = min(k, len(valid))
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            encoder = SentenceTransformer(self.embedding_model)
            train_queries = [get_question(d["input"]) for d in valid]
            embs = encoder.encode(train_queries, convert_to_numpy=True)
            q_emb = encoder.encode([query], convert_to_numpy=True)[0]
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            normed = embs / (norms + 1e-9)
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            sims = normed @ q_norm
            top_idx = np.argsort(sims)[::-1][:k]
            return [
                (get_question(valid[int(i)]["input"]), valid[int(i)]["entry"]["program"])
                for i in top_idx
            ]
        except ImportError:
            return [
                (get_question(d["input"]), d["entry"]["program"])
                for d in valid[:k]
            ]

    @staticmethod
    def _extract_program(response: str) -> str:
        """
        Extract program code from agent response.
        Strips markdown fences and leading 'Thought and Program:' prefixes.
        """
        text = response.strip()
        # Strip markdown code fences
        if "```python" in text:
            start = text.index("```python") + len("```python")
            end = text.index("```", start) if "```" in text[start:] else len(text)
            return text[start:end].strip()
        if text.startswith("```"):
            start = text.index("\n") + 1 if "\n" in text else 3
            end = text.rfind("```")
            return text[start:end].strip() if end > start else text[start:].strip()
        return text

    # ------------------------------------------------------------------
    # solve_with_reward: one-shot wrapper for main.py compatibility
    # ------------------------------------------------------------------

    def solve_with_reward(
        self,
        task_input: dict,
        task_type: str,
        budget: float,
        reward_fn: Callable,
        entry: dict,
        max_reward_iters: int = 1,
    ) -> dict:
        """
        One-shot solve + reward evaluation for main.py compatibility.

        ReGAL has no online retry loop (training is offline); this wrapper
        calls solve() once and evaluates the reward.
        """
        result = self.solve(task_input, task_type)
        answer = result["final_output"].get("answer", "")

        try:
            reward, feedback = reward_fn(answer, entry)
        except Exception as exc:
            logger.debug("reward_fn failed: %s", exc)
            reward, feedback = 0.0, str(exc)

        result["best_reward"] = reward
        result["final_reward"] = reward
        result["reward_history"] = [
            {
                "iteration": 0,
                "reward": reward,
                "blame": "regal",
                "message": str(feedback)[:120] if feedback else "",
            }
        ]
        result["solved"] = reward >= 1.0

        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, codebank_path: str, demobank_path: str) -> None:
        self.codebank.save(codebank_path)
        self.demobank.save(demobank_path)
        logger.info("Saved CodeBank → %s, DemoBank → %s", codebank_path, demobank_path)

    def load(self, codebank_path: str, demobank_path: str) -> None:
        self.codebank = ReGALCodeBank.load(codebank_path)
        self.demobank = ReGALDemoBank.load(demobank_path)
        logger.info("Loaded CodeBank ← %s, DemoBank ← %s", codebank_path, demobank_path)

    # ------------------------------------------------------------------
    # Stats (for --stats flag in main.py)
    # ------------------------------------------------------------------

    def library_stats(self) -> dict:
        funcs = []
        for f in self.codebank.functions:
            score, n = f.compute_success()
            funcs.append({
                "name": f.name,
                "usage_count": n,
                "score": round(score, 3),
                "round_added": f.round_added,
                "description": f.description,
            })
        return {
            "num_functions": len(self.codebank),
            "num_demos": len(self.demobank),
            "functions": funcs,
            "cost_summary": {
                "framework": "regal",
                "codebank_size": len(self.codebank),
                "demobank_size": len(self.demobank),
            },
            "cost_log": [],
        }
