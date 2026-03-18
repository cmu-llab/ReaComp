import os
import json
import reasoning_gym
import random

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
    "composite": False,
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

import reasoning_gym
import random


def build_rg_split_with_difficulty(
    N=20,
    f=0.25,
    seed=42,
    difficulty=None,  # can be scalar or callable(task)->difficulty
    shuffle=True,
    skip_failures=True,
):
    rng = random.Random(seed)

    train_data = []
    test_data = []

    test_size = max(1, int(f * N))

    for i, name in enumerate(get_all_datasets()):
        train_seed = seed + i * 2
        test_seed = seed + i * 2 + 1

        kwargs = {}

        # inject difficulty ONLY if supported
        if has_curriculum(name) and difficulty is not None:
            if callable(difficulty):
                kwargs["difficulty"] = difficulty(name)
            else:
                kwargs["difficulty"] = difficulty

        try:
            train_ds = reasoning_gym.create_dataset(
                name, size=N, seed=train_seed, **kwargs
            )
            test_ds = reasoning_gym.create_dataset(
                name, size=test_size, seed=test_seed, **kwargs
            )
        except Exception as e:
            if skip_failures:
                print(f"[WARN] Skipping {name}: {e}")
                continue
            else:
                raise

        for x in train_ds:
            x["task"] = name
            x["has_curriculum"] = has_curriculum(name)

        for x in test_ds:
            x["task"] = name
            x["has_curriculum"] = has_curriculum(name)

        train_data.extend(train_ds)
        test_data.extend(test_ds)

    if shuffle:
        rng.shuffle(train_data)
        rng.shuffle(test_data)

    return train_data, test_data

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

    for i, name in enumerate(datasets):
        train_seed = seed + i * 2
        test_seed = seed + i * 2 + 1

        try:
            train_ds = reasoning_gym.create_dataset(
                name, size=N, seed=train_seed
            )
            test_ds = reasoning_gym.create_dataset(
                name, size=test_size, seed=test_seed
            )
        except Exception as e:
            if skip_failures:
                print(f"[WARN] Skipping {name}: {e}")
                continue
            else:
                raise

        # annotate task name
        for x in train_ds:
            x["task"] = name
        for x in test_ds:
            x["task"] = name

        train_data.extend(train_ds)
        test_data.extend(test_ds)

    if shuffle:
        rng.shuffle(train_data)
        rng.shuffle(test_data)

    return train_data, test_data

# main
if __name__ == "__main__":
    train, test = build_rg_split(N=20, f=0.25, seed=42, shuffle=True)

    print(len(train))  # ~2000
    print(len(test))   # ~500