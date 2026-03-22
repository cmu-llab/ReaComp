import os
import sys
import json
import string
import pathlib
from enum import Enum
from tqdm import tqdm

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.data_generation.utils import read_jsonl

class ProgramVocabulary(set):
    """The set of characters used for creating predicate (input) and 
    transform (output) windows where a given program operates
    """
    def __init__(self, *args, **kwargs):
        if len(args) > 2: # just some syntax sugar.
            super().__init__(list(args))
        else: super().__init__(*args, **kwargs)

    def __str__(self):
        return "".join(list(self))
    
    def __repr__(self):
        return "".join(list(self))

    def has_word(self, word: str):
        word_vocab = set(list(word.strip()))
        
        return word_vocab.issubset(self)

# Possible edges/relations between programs.
class ProgramBFCCEdge(Enum):
    N = "N" #"Neutral"
    F = "F" #'Feeding'
    B = "B" #'Bleeding'

class ProgramBFCCNode:
    """Abstract representation of program in the BFCC interaction graph."""
    def __init__(self, predicate: str, transform: str, vocabulary: ProgramVocabulary=ProgramVocabulary('a','b','c','d','e','f'), max_window_size: int=3):
        self.predicate = predicate
        self.transform = transform
        self.vocabulary = vocabulary
        self.max_window_size = max_window_size

    @property
    def alpha(self):
        return self.predicate
    
    @property
    def beta(self):
        return self.transform

    @classmethod
    def from_string(cls, string_repr: str, max_window_size: int=3, vocabulary: ProgramVocabulary=ProgramVocabulary('a','b','c','d','e','f')):
        self = cls("","")
        self.vocabulary = vocabulary
        self.max_window_size = max_window_size
        assert string_repr.strip() != '', "passed blank program input"
        assert 'replace(' in string_repr, f"replace function not found in program string: {string_repr}"
        
        # extract predicate and transform
        predicate = string_repr.split("replace(")[-1].split(",")[0].strip()
        transform = string_repr.split("replace(")[-1].split(",")[1].split(")")[0].strip()

        # strip quotations off
        predicate = predicate.replace("'","").replace('"','')
        transform = transform.replace("'","").replace('"','')
        # print("predicate:", predicate)
        # print("transform:", transform)
        
        # restrict the length of predicate and transform windows (no. of characters in them).
        assert len(predicate) <= self.max_window_size, f"predicate has more than {self.max_window_size} characters: {predicate} in ({string_repr})"
        assert len(transform) <= self.max_window_size, f"transform has more than {self.max_window_size} characters: {transform} in ({string_repr})"
        
        # restrict the vocabulary of predicates and transforms.
        assert self.vocabulary.has_word(predicate), f"the predicate {predicate} has some symbols not in the vocabulary: {self.vocabulary}" 
        assert self.vocabulary.has_word(transform), f"the transform {transform} has some symbols not in the vocabulary: {self.vocabulary}" 

        self.predicate = predicate
        self.transform = transform

        return self
    
    def __call__(self, inputs: list[str]) -> list[str]:
        outputs: list[str] = []
        for input in inputs:
            output = input.replace(self.predicate, self.transform)
            outputs.append(output)

        return outputs

    def __eq__(self, other):
        return self.predicate == other.predicate and self.transform == other.transform

    def __repr__(self):
        return f'replace("{self.predicate}", "{self.transform}")'
    
    def __str__(self):
        return self.__repr__()

class ProgramBFCCInteractionDAG:
    def __init__(self, triples):
        self.triples

# main
if __name__ == "__main__":
    data = read_jsonl('data/example_data.jsonl')
    # validating data.
    for instance in data:
        for program in instance["programs"]:
            # check if a valid program can be constructed from string outputs.
            program_node = ProgramBFCCNode.from_string(program, vocabulary="abcdefghijkuvwxyz")
            print(program_node.predicate)
            print(program_node.transform)
            # TODO: add checks for BFCC DAG.