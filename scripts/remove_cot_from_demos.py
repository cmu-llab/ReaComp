import json
import random

# main
if __name__ == "__main__":
    data = json.load(open("demos/DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json", "r"))
    hard_success = [rec for rec in data if rec["difficulty"] == "hard" and rec["success"] == True]
    hard_failure = [rec for rec in data if rec["difficulty"] == "hard" and rec["success"] == False]
    easy_success = [rec for rec in data if rec["difficulty"] == "easy" and rec["success"] == True]
    easy_failure = [rec for rec in data if rec["difficulty"] == "easy" and rec["success"] == False]

    random.seed(42)
    examples_48_set = random.sample(hard_success, 12) + random.sample(hard_failure, 12) + random.sample(easy_success, 12) + random.sample(easy_failure, 12)
    examples_12_set = random.sample(hard_success, 3) + random.sample(hard_failure, 3) + random.sample(easy_success, 3) + random.sample(easy_failure, 3)

    json.dump(examples_48_set, open("demos/DEMOS_PBEBENCH_seed_42_48_examples.json", "w"))
    json.dump(examples_12_set, open("demos/DEMOS_PBEBENCH_seed_42_12_examples.json", "w"))
    
    for rec in data:
        rec.pop("cot", None)
    json.dump(data, open("demos/DEMOS_PBEBENCH_seed_42_100_examples.json", "w"))