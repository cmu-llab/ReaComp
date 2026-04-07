import os
import sys
import pathlib
import datasets

sys.path.append(pathlib.Path(str(os.path.abspath(__file__))).parent)

from utils import write_jsonl

OLD_PROMPT_TEMPLATE = """Follow the instructions below to solve the code completion task:

We will provide the input corpus and corresponding output corpus. Each element in the corpus is a string, and the output is transformed from the corresponding input using an ordered sequence of "replace" programs. You need to find the correctly constructed and ordered sequence of "replace" programs to transform the entire input corpus into the output corpus. Note that the programs can interact with each other in a way that reduces or increases the number of times they are applied on a given input based on where they are ordered in the sequence. This makes it very important to apply them in the correct order.

The programs should be written using only the Python replace function. For example, for a program that replaces all occurrences of "ab" with "bc" it should be written as: ```replace(‘ab’, ‘bc’)```

Here is an example of the full task:
### Inputs
["abc", "ebc", "aba"]

### Outputs
["edc", "edc", "aba"]

### Program Sequence
```python
["replace(‘bc’,’dc’)", "replace(‘ad’,’ed’)"]
```

While generating the program sequence, you need to abide by the following restrictions:
1. Each program in the sequence should have the form "replace(A, B)", where A and B are both strings.
2. Both argument strings A and B in "replace(A, B)" should have <= {program_length} characters. A should have at least 1 character but B can be null (or "").
3. The maximum number of programs in a sequence is {program_num}
4. You should only consider the Python ‘replace’ function for specifying programs (each program is a Python replace function). You can not use any other Python modules or functions.
5. Strictly follow the markdown style convention while presenting your final program sequence, and make sure to enclose it in the ```python``` markdown style code block.

Now, please generate the sequence of programs corresponding to the following input corpus and output corpus:

### Inputs
{inputs_list}

### Outputs
{outputs_list}

### Program Sequence
"""

PROMPT_TEMPLATE = """\
Given the input/output pairs below, find the ordered sequence of Python str.replace() calls \
that transforms every input into its output.

Inputs:  {inputs_list}
Outputs: {outputs_list}

Constraints:
- Each program must have the form replace(A, B) where A and B are strings.
- len(A) must be between 1 and {program_length} characters; len(B) must be between 0 and {program_length} characters.
- At most {program_num} programs in the sequence.
- Only the Python str.replace() function may be used.
- Your answer must be a Python list of replace() call strings, e.g. ["replace('ab','c')", "replace('d','ef')"].\
"""

def _build_prompt(rec: dict, program_length: int = 3, program_num: int = 5) -> str:
    """Build a clean prompt for a PBEBench record using PROMPT_TEMPLATE."""
    return PROMPT_TEMPLATE.format(
        inputs_list=rec["inputs"],
        outputs_list=rec["outputs"],
        program_length=program_length,
        program_num=program_num,
    )


def load_pbebench_lite(path: str="changelinglab/PBEBench-Lite", first_k: int=-1):
    """Load the PBEBench-Lite dataset from Hugging Face. If `first_k` is set to a positive integer, only the first `k` records will be returned."""
    ds = datasets.load_dataset(path)['test']
    sel_data = []
    for rec in ds:
        rec = dict(rec)
        rec['prompt'] = _build_prompt(rec)
        rec['reward'] = "pbebench"
        sel_data.append(rec)

    if first_k > 0: sel_data = sel_data[:first_k]

    return sel_data

# main
if __name__ == "__main__":
    # data = load_pbebench_lite(first_k=50)
    data = load_pbebench_lite(first_k=-1)
    write_path = "data/pbebench/lite_tasks_full.jsonl"
    write_jsonl(data, write_path)