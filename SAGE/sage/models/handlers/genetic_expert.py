"""
Genetic expert handler class

Copyright (c) 2022 Hocheol Lim.
"""

import gc
import random
from typing import List

import numpy as np
import torch
from joblib import Parallel, delayed
from rdkit import Chem

from sage.models.expert import (
    crossover,
    mutate,
)


class GeneticOperatorHandler:
    def __init__(
        self,
        crossover_type: str,
        mutation_type: str,
        mutation_initial_rate: float,
    ) -> None:
        self.mutation_initial_rate = mutation_initial_rate
        self.mutation_rate = mutation_initial_rate

        if crossover_type == "SMILES":
            self.crossover_func = crossover
        else:
            raise ValueError(f"'crossover_type' {crossover_type} is invalid")

        if mutation_type == "SMILES":
            self.mutate_func = mutate
        else:
            raise ValueError(f"'mutation_type' {mutation_type} is invalid")

    def query(
        self,
        query_size: int,
        apprentice_mean_similarity: float,
        mating_pool: List[str],
        pool: Parallel,
    ) -> List[str]:

        original_smiles = random.choices(mating_pool, k=2 * query_size)
        smiles_a, smiles_b = (
            original_smiles[:query_size],
            original_smiles[query_size:],
        )
        smiles = pool(
            delayed(self.reproduce_mols)(smile_a, smile_b, self.mutation_rate)
            for smile_a, smile_b in zip(smiles_a, smiles_b)
        )

        smiles_list = list(filter(lambda smile: smile is not None, smiles))
        gc.collect()
        return smiles_list

    def reproduce_mols(
        self, parent_a: str, parent_b: str, mutation_rate: float
    ) -> List[str]:
        new_child = self.crossover_func(parent_a, parent_b)
        if new_child is not None:
            new_child = self.mutate_func(new_child, mutation_rate)

        smiles = (
            Chem.MolToSmiles(new_child, isomericSmiles=True)
            if new_child is not None
            else None
        )
        return smiles

    def reproduce_frags(self, smiles_list: List[str], mutation_rate: float) -> str:
        num_fragments = np.random.randint(2, 6)
        fragments = np.random.choice(smiles_list, num_fragments, replace=True).tolist()

        fragments_mol = [Chem.MolFromSmiles(frag) for frag in fragments]
        new_child = self.crossover_func(fragments_mol)  # type: ignore
        if new_child is not None:
            new_child = self.mutate_func(new_child, mutation_rate)

        smiles = (
            Chem.MolToSmiles(new_child, isomericSmiles=True)
            if new_child is not None
            else None
        )
        return smiles

