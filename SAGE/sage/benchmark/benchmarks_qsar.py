"""
General Tasks for Optimization
Copyright (c) 2022 Hocheol Lim.
"""

import sys
import os
from rdkit.Chem import RDConfig

import networkx as nx
from sage.scoring.common_scoring_functions import (
    CNS_MPO_ScoringFunction,
    IsomerScoringFunction,
    RdkitScoringFunction,
    SMARTSScoringFunction,
    TanimotoScoringFunction,
)
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
from sage.scoring.score_modifier import (
    ClippedScoreModifier,
    GaussianModifier,
    MaxGaussianModifier,
    MinGaussianModifier,
)
from sage.scoring.scoring_function import (
    ArithmeticMeanScoringFunction,
    GeometricMeanScoringFunction,
    ScoringFunction,
    MoleculewiseScoringFunction,
)

from sage.scoring.qsar_models import (
    AChE_score_1_score,
    COX2_score_1_score,
    PKCB_score_1_score,
    FGFR1_score_1_score,
    PTP1B_score_1_score,
    
    AChE_score_2_score,
    COX2_score_2_score,
    PKCB_score_2_score,
    FGFR1_score_2_score,
    PTP1B_score_2_score,

    AChE_score_3_score,
    COX2_score_3_score,
    PKCB_score_3_score,
    FGFR1_score_3_score,
    PTP1B_score_3_score,
    
    AChE_score_4_score,
    COX2_score_4_score,
    PKCB_score_4_score,
    FGFR1_score_4_score,
    PTP1B_score_4_score,
    
    MAOB_score_1_score,
    MAOB_score_2_score,
    MAOB_score_3_score,
    MAOB_score_4_score,
    
    AChE_MAOB_score_1_score,
    AChE_MAOB_score_2_score,
    AChE_MAOB_score_3_score,
    AChE_MAOB_score_4_score,
)

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol


class ThresholdedImprovementScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, objective, constraint, threshold, offset):
        super().__init__()
        self.objective = objective
        self.constraint = constraint
        self.threshold = threshold
        self.offset = offset

    def raw_score(self, smiles):
        score = (
            self.corrupt_score
            if (self.constraint.score(smiles) < self.threshold)
            else (self.objective.score(smiles) + self.offset)
        )
        return score

def AChE_score_1_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="AChE score 1",
        objective=RdkitScoringFunction(descriptor=AChE_score_1_score),
        contribution_specification=specification,
    )

def COX2_score_1_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="COX2 score 1",
        objective=RdkitScoringFunction(descriptor=COX2_score_1_score),
        contribution_specification=specification,
    )

def PKCB_score_1_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="PKCB score 1",
        objective=RdkitScoringFunction(descriptor=PKCB_score_1_score),
        contribution_specification=specification,
    )

def FGFR1_score_1_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="FGFR1 score 1",
        objective=RdkitScoringFunction(descriptor=FGFR1_score_1_score),
        contribution_specification=specification,
    )

def PTP1B_score_1_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="PTP1B score 1",
        objective=RdkitScoringFunction(descriptor=PTP1B_score_1_score),
        contribution_specification=specification,
    )

def AChE_score_2_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="AChE score 2",
        objective=RdkitScoringFunction(descriptor=AChE_score_2_score),
        contribution_specification=specification,
    )

def COX2_score_2_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="COX2 score 2",
        objective=RdkitScoringFunction(descriptor=COX2_score_2_score),
        contribution_specification=specification,
    )

def PKCB_score_2_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="PKCB score 2",
        objective=RdkitScoringFunction(descriptor=PKCB_score_2_score),
        contribution_specification=specification,
    )

def FGFR1_score_2_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="FGFR1 score 2",
        objective=RdkitScoringFunction(descriptor=FGFR1_score_2_score),
        contribution_specification=specification,
    )

def PTP1B_score_2_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="PTP1B score 2",
        objective=RdkitScoringFunction(descriptor=PTP1B_score_2_score),
        contribution_specification=specification,
    )

def AChE_score_3_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="AChE score 3",
        objective=RdkitScoringFunction(descriptor=AChE_score_3_score),
        contribution_specification=specification,
    )

def COX2_score_3_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="COX2 score 3",
        objective=RdkitScoringFunction(descriptor=COX2_score_3_score),
        contribution_specification=specification,
    )

def PKCB_score_3_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="PKCB score ",
        objective=RdkitScoringFunction(descriptor=PKCB_score_3_score),
        contribution_specification=specification,
    )

def FGFR1_score_3_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="FGFR1 score 3",
        objective=RdkitScoringFunction(descriptor=FGFR1_score_3_score),
        contribution_specification=specification,
    )

def PTP1B_score_3_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="PTP1B score 3",
        objective=RdkitScoringFunction(descriptor=PTP1B_score_3_score),
        contribution_specification=specification,
    )

def AChE_score_4_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="AChE score 4",
        objective=RdkitScoringFunction(descriptor=AChE_score_4_score),
        contribution_specification=specification,
    )

def COX2_score_4_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="COX2 score 4",
        objective=RdkitScoringFunction(descriptor=COX2_score_4_score),
        contribution_specification=specification,
    )

def PKCB_score_4_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="PKCB score 4",
        objective=RdkitScoringFunction(descriptor=PKCB_score_4_score),
        contribution_specification=specification,
    )

def FGFR1_score_4_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="FGFR1 score 4",
        objective=RdkitScoringFunction(descriptor=FGFR1_score_4_score),
        contribution_specification=specification,
    )

def PTP1B_score_4_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="PTP1B score 4",
        objective=RdkitScoringFunction(descriptor=PTP1B_score_4_score),
        contribution_specification=specification,
    )

def MAOB_score_1_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="MAOB score 1",
        objective=RdkitScoringFunction(descriptor=MAOB_score_1_score),
        contribution_specification=specification,
    )

def MAOB_score_2_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="MAOB score 2",
        objective=RdkitScoringFunction(descriptor=MAOB_score_2_score),
        contribution_specification=specification,
    )

def MAOB_score_3_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="MAOB score 3",
        objective=RdkitScoringFunction(descriptor=MAOB_score_3_score),
        contribution_specification=specification,
    )

def MAOB_score_4_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="MAOB score 4",
        objective=RdkitScoringFunction(descriptor=MAOB_score_4_score),
        contribution_specification=specification,
    )

def AChE_MAOB_score_1_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="AChE_MAOB score 1",
        objective=RdkitScoringFunction(descriptor=AChE_MAOB_score_1_score),
        contribution_specification=specification,
    )

def AChE_MAOB_score_2_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="AChE_MAOB score 2",
        objective=RdkitScoringFunction(descriptor=AChE_MAOB_score_2_score),
        contribution_specification=specification,
    )

def AChE_MAOB_score_3_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="AChE_MAOB score 3",
        objective=RdkitScoringFunction(descriptor=AChE_MAOB_score_3_score),
        contribution_specification=specification,
    )

def AChE_MAOB_score_4_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="AChE_MAOB score 4",
        objective=RdkitScoringFunction(descriptor=AChE_MAOB_score_4_score),
        contribution_specification=specification,
    )