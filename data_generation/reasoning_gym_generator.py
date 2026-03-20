import os
import reasoning_gym
import random
from tqdm import tqdm
from fractions import Fraction
from .utils import write_jsonl

DATASET_METADATA = {
    "ab": True,
    "acre": False,
    "advanced_geometry": True,
    "aiw": True,
    "arc_1d": True,
    "arc_agi": True,
    "base_conversion": True,
    "basic_arithmetic": True,
    "bf": True,
    "binary_alternation": True,
    "binary_matrix": True,
    "bitwise_arithmetic": True,
    "boxnet": True,
    "caesar_cipher": True,
    "calendar_arithmetic": True,
    "chain_sum": True,
    "circuit_logic": True,
    "codeio": True,
    "coin_flip": True,
    "color_cube_rotation": True,
    "complex_arithmetic": True,
    # "composite": False,
    "count_bits": True,
    "count_primes": True,
    "countdown": True,
    "course_schedule": True,
    "cryptarithm": True,
    "decimal_arithmetic": True,
    "decimal_chain_sum": True,
    "dice": True,
    "emoji_mystery": True,
    "family_relationships": True,
    "figlet_font": True,
    "fraction_simplification": True,
    "futoshiki": True,
    "game_of_life": True,
    "game_of_life_halting": True,
    "gcd": True,
    "graph_color": True,
    "group_anagrams": True,
    "gsm_symbolic": False,
    "intermediate_integration": True,
    "isomorphic_strings": True,
    "jugs": True,
    "kakurasu": True,
    "knight_swap": True,
    "knights_knaves": True,
    "largest_island": True,
    "lcm": True,
    "leg_counting": True,
    "letter_counting": True,
    "letter_jumble": True,
    "list_functions": False,
    "mahjong_puzzle": True,
    "manipulate_matrix": True,
    "maze": True,
    "mini_sudoku": True,
    "modulo_grid": True,
    "n_queens": True,
    "needle_haystack": True,
    "number_filtering": True,
    "number_format": True,
    "number_sequence": True,
    "number_sorting": True,
    "palindrome_generation": True,
    "palindrome_partitioning": True,
    "polynomial_equations": True,
    "polynomial_multiplication": True,
    "pool_matrix": True,
    "power_function": True,
    "prime_factorization": True,
    "products": True,
    "propositional_logic": True,
    "puzzle24": True,
    "quantum_lock": True,
    "ransom_note": True,
    "rearc": True,
    "rectangle_count": True,
    "rotate_matrix": True,
    "rotten_oranges": True,
    "rubiks_cube": True,
    "rush_hour": True,
    "self_reference": True,
    "sentence_reordering": True,
    "shortest_path": True,
    "simple_equations": True,
    "simple_geometry": True,
    "simple_integration": True,
    "sokoban": True,
    "spell_backward": True,
    "spiral_matrix": True,
    "string_insertion": True,
    "string_manipulation": True,
    "string_splitting": True,
    "string_synthesis": True,
    "sudoku": True,
    "survo": True,
    "syllogism": True,
    "time_intervals": True,
    "tower_of_hanoi": True,
    "tsumego": True,
    "word_ladder": True,
    "word_sequence_reversal": True,
    "word_sorting": True,
    "zebra_puzzles": True,
}

def has_curriculum(task_name):
    return DATASET_METADATA.get(task_name, False)


def get_all_datasets():
    return list(DATASET_METADATA.keys())


def get_curriculum_datasets():
    return [k for k, v in DATASET_METADATA.items() if v]


def get_non_curriculum_datasets():
    return [k for k, v in DATASET_METADATA.items() if not v]

# Your dataset list (you can keep this in a separate config file)
ALL_DATASETS = [
    "ab","acre","advanced_geometry","aiw","arc_1d","arc_agi","base_conversion",
    "basic_arithmetic","bf","binary_alternation","binary_matrix","bitwise_arithmetic",
    "boxnet","caesar_cipher","calendar_arithmetic","chain_sum","circuit_logic",
    "codeio","coin_flip","color_cube_rotation","complex_arithmetic",
    "count_bits","count_primes","countdown","course_schedule","cryptarithm",
    "decimal_arithmetic","decimal_chain_sum","dice","emoji_mystery",
    "family_relationships","figlet_font","fraction_simplification","futoshiki",
    "game_of_life","game_of_life_halting","gcd","graph_color","group_anagrams",
    "gsm_symbolic","intermediate_integration","isomorphic_strings","jugs",
    "kakurasu","knight_swap","knights_knaves","largest_island","lcm",
    "leg_counting","letter_counting","letter_jumble","list_functions",
    "mahjong_puzzle","manipulate_matrix","maze","mini_sudoku","modulo_grid",
    "n_queens","needle_haystack","number_filtering","number_format",
    "number_sequence","number_sorting","palindrome_generation",
    "palindrome_partitioning","polynomial_equations",
    "polynomial_multiplication","pool_matrix","power_function",
    "prime_factorization","products","propositional_logic","puzzle24",
    "quantum_lock","ransom_note","rearc","rectangle_count","rotate_matrix",
    "rotten_oranges","rubiks_cube","rush_hour","self_reference",
    "sentence_reordering","shortest_path","simple_equations",
    "simple_geometry","simple_integration","sokoban","spell_backward",
    "spiral_matrix","string_insertion","string_manipulation",
    "string_splitting","string_synthesis","sudoku","survo","syllogism",
    "time_intervals","tower_of_hanoi","tsumego","word_ladder",
    "word_sequence_reversal","word_sorting","zebra_puzzles"
]

def seralize_variable(value):
    if isinstance(value, Fraction):
        return repr(value)
    return value

def serialize_gsm_symbolic_variables(rec: dict) -> dict:
    """ GSM symbolic example (from reasoning-gym/datasets/gsm_symbolic.py) sometimes has non-serializable variables (e.g. Fraction), so we convert them to strings for JSON serialization.
    {
        'question': 'A hospital has a capacity of 5600 wards with 1/4 occupied. Due to the pandemic, 90 patients are admitted into the hospital each day. Calculate the total number of unoccupied wards in the hospital after 4 weeks. Give the result as your final answer. Do not include units.', 
        'answer': '1680', 
        'metadata': {
            'difficulty': 1.0, 
            'answer_value': 1680, 
            'answer_cot': 'If 1/4 of the total capacity of the hospital wards is occupied, it means 1/4 * 5600 = 1400 wards have patients using them.\nThe total number of wards in the hospital without new admissions is 5600 wards - 1400 wards = 4200 wards.\nIf 90 people are admitted each day, the total number of patients in the hospital after one week is 90 patients/day * 7 days/week = 630 patients.\nAfter 4 weeks, the total number of patients admitted into the hospital is 630 patients/week * 4 weeks = 2520 patients, who each use one ward.\nIf there were 4200 unoccupied wards in the hospital before the new admissions, the total number is reduced to 4200 wards - 2520 wards = 1680 unoccupied wards.\n#### 1680', 
            'variables': {
                'facility': 'hospital', 
                'total_capacity': 5600, 
                'item': 'ward', 
                'initial_fraction': Fraction(1, 4), 
                'event': 'pandemic', 
                'daily_patients': 90, 
                'period_weeks': 4, 
                'initial_occupied': 1400, 
                'initial_empty': 4200, 
                'total_admitted': 2520
            }, 
            'source_dataset': 'gsm_symbolic', 
            'source_index': 21
        }, 
        'task': 'gsm_symbolic'
    }
    """
    rec['metadata']['variables'] = {k: seralize_variable(v) for k, v in rec['metadata']['variables'].items()}

    return rec

def build_rg_split(
    N=20,
    f=0.25,
    seed=42,
    datasets=ALL_DATASETS,
    shuffle=True,
    skip_failures=True,
):
    """
    Build train/test splits across reasoning-gym datasets.

    Args:
        N: train samples per dataset
        f: test fraction (test = f * N)
        seed: global seed
        datasets: list of dataset names
        shuffle: shuffle final outputs
        skip_failures: skip datasets that error out

    Returns:
        train_data, test_data
    """
    rng = random.Random(seed)

    train_data = []
    test_data = []

    test_size = max(1, int(f * N))
    pbar = tqdm(enumerate(datasets), total=len(datasets), desc="Building RG split")

    for i, name in pbar:
        train_seed = seed + i * 2
        test_seed = seed + i * 2 + 1

        try:
            train_ds_and_eg = list(reasoning_gym.create_dataset(
                name, size=N+1, seed=train_seed
            ))
            train_ds = train_ds_and_eg[:-1]  # all but last example for training
            eg = train_ds_and_eg[-1]  # last example for prompt
            if f > 0:
                test_ds = list(reasoning_gym.create_dataset(
                    name, size=test_size, seed=test_seed
                ))
            else: test_ds = []
        except Exception as e:
            if skip_failures:
                print(f"[WARN] Skipping {name}: {e}")
                continue
            else:
                raise

        # annotate task name
        for x in train_ds:
            x["task"] = name
            x["prompt"] = f"""You will be shown an example reasoning problem below.
Question: {eg['question']}
Answer: {eg['answer']}

Now solve a similar reasoning problem and report the final answer in a similar format as above. Just give the answer, without any explanation.

Question: {x['question']}
Answer:"""
            x = serialize_gsm_symbolic_variables(x) if name == "gsm_symbolic" else x
        for x in test_ds:
            x["task"] = name
            x["prompt"] = f"""You will be shown an example reasoning problem below.
Question: {eg['question']}
Answer: {eg['answer']}

Now solve a similar reasoning problem and report the final answer in a similar format as above. Just give the answer, without any explanation.

Question: {x['question']}
Answer:"""
            x = serialize_gsm_symbolic_variables(x) if name == "gsm_symbolic" else x

        train_data.extend(train_ds)
        test_data.extend(test_ds)

        # # TODO: DEBUG - check for JSON serialization issues immediately after dataset creation, to identify which dataset(s) cause issues
        # for rec in train_ds:
        #     try: json.dumps(rec)
        #     except TypeError as e:
        #         print(rec, train_seed)
        #         print(f"Serialization error in train_ds for task {rec['task']}: {e}")
        #         exit()

    if shuffle:
        rng.shuffle(train_data)
        if len(test_data) > 0: 
            rng.shuffle(test_data)

    return train_data, test_data

def create_pilot_run_data():
    easy_pilot_tasks_path = "data/reasoning_gym/easy_pilot_tasks.jsonl"
    med_pilot_tasks_path = "data/reasoning_gym/med_pilot_tasks.jsonl"
    hard_pilot_tasks_path = "data/reasoning_gym/hard_pilot_tasks.jsonl"

    easy_test, _ = build_rg_split(N=50, f=0, seed=42, datasets=["basic_arithmetic", "caesar_cipher"], shuffle=False) # easy tasks.
    if not os.path.exists(easy_pilot_tasks_path):
        write_jsonl(easy_test, easy_pilot_tasks_path)

    med_test, _ = build_rg_split(N=50, f=0, seed=42, datasets=["palindrome_partitioning",
  "calendar_arithmetic"], shuffle=False) # medium tasks.
    if not os.path.exists(med_pilot_tasks_path):
        write_jsonl(med_test, med_pilot_tasks_path)

    hard_test, _ = build_rg_split(N=50, f=0, seed=42, datasets=["sokoban", "arc_agi"], shuffle=False) # hard tasks.
    if not os.path.exists(hard_pilot_tasks_path):
        write_jsonl(hard_test, hard_pilot_tasks_path)

def create_final_data():
    train_path = "data/reasoning_gym/train.jsonl"
    test_path = "data/reasoning_gym/test.jsonl"
    train, test = build_rg_split(N=100, f=0.25, seed=42, shuffle=True)

    print(len(train))  # ~10000
    print(len(test))   # ~2500
    if not os.path.exists(train_path):
        write_jsonl(train, train_path)
    if not os.path.exists(test_path):
        write_jsonl(test, test_path)

# main
if __name__ == "__main__":
    create_pilot_run_data()
    # create_final_data()