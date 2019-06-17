import numpy as np
import pickle
from itertools import permutations


def write_mapping():
    class_mapping = { (0, 0, 0, 0): 0}
    idx = 1

    possibilities = [
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 1, 1, 0),
        (1, 1, 1, 1)
            ]

    for p in possibilities:
        perm = permutations(p)
        set_perm = set([i for i in list(perm)])
        for perm in set_perm:
            class_mapping[perm] = idx
            idx += 1

    with open("class_mapping.pkl", "wb") as f:
        pickle.dump(class_mapping, f)

def read_mapping(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
