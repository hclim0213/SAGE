"""
Benchmark loader methods
Copyright (c) 2022 Hocheol Lim.
"""

from typing import List, Tuple

from guacamol.goal_directed_benchmark import GoalDirectedBenchmark

from sage.benchmark.benchmarks_gegl import (
    penalized_logp_atomrings,
    penalized_logp_cyclebasis,
    amlodipine_rings,
    cns_mpo,
    decoration_hop,
    hard_fexofenadine,
    hard_osimertinib,
    isomers_c7h8n2o2,
    isomers_c9h10n2o2pf2cl,
    isomers_c11h24,
    logP_benchmark,
    median_camphor_menthol,
    median_tadalafil_sildenafil,
    perindopril_rings,
    pioglitazone_mpo,
    qed_benchmark,
    ranolazine_mpo,
    scaffold_hop,
    similarity,
    sitagliptin_replacement,
    tpsa_benchmark,
    valsartan_smarts,
    zaleplon_with_other_formula,
)

from sage.benchmark.benchmarks_qsar import ((
    AChE_score_1_task,
    COX2_score_1_task,
    PKCB_score_1_task,
    FGFR1_score_1_task,
    PTP1B_score_1_task,
    AChE_score_2_task,
    COX2_score_2_task,
    PKCB_score_2_task,
    FGFR1_score_2_task,
    PTP1B_score_2_task,
    AChE_score_3_task,
    COX2_score_3_task,
    PKCB_score_3_task,
    FGFR1_score_3_task,
    PTP1B_score_3_task,
    AChE_score_4_task,
    COX2_score_4_task,
    PKCB_score_4_task,
    FGFR1_score_4_task,
    PTP1B_score_4_task,
    MAOB_score_1_task,
    MAOB_score_2_task,
    MAOB_score_3_task,
    MAOB_score_4_task,
    AChE_MAOB_score_1_task,
    AChE_MAOB_score_2_task,
    AChE_MAOB_score_3_task,
    AChE_MAOB_score_4_task,
)

def load_benchmark(benchmark_id: int) -> Tuple[GoalDirectedBenchmark, List[int]]:
    benchmark = {
        0: similarity(
            smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
            name="Celecoxib",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        1: similarity(
            smiles="Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
            name="Troglitazone",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        2: similarity(
            smiles="CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
            name="Thiothixene",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        3: similarity(
            smiles="Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl",
            name="Aripiprazole",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        4: similarity(
            smiles="CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
            name="Albuterol",
            fp_type="FCFP4",
            threshold=0.75,
        ),
        5: similarity(
            smiles="COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
            name="Mestranol",
            fp_type="AP",
            threshold=0.75,
        ),
        6: isomers_c11h24(),
        7: isomers_c9h10n2o2pf2cl(),
        8: median_camphor_menthol(),
        9: median_tadalafil_sildenafil(),
        10: hard_osimertinib(),
        11: hard_fexofenadine(),
        12: ranolazine_mpo(),
        13: perindopril_rings(),
        14: amlodipine_rings(),
        15: sitagliptin_replacement(),
        16: zaleplon_with_other_formula(),
        17: valsartan_smarts(),
        18: decoration_hop(),
        19: scaffold_hop(),
        20: logP_benchmark(target=-1.0),
        21: logP_benchmark(target=8.0),
        22: tpsa_benchmark(target=150.0),
        23: cns_mpo(),
        24: qed_benchmark(),
        25: isomers_c7h8n2o2(),
        26: pioglitazone_mpo(),
        27: penalized_logp_atomrings(),
        28: penalized_logp_cyclebasis(),
        
        # Bridged bicyclic ring
        29: similarity(
            smiles="CC=C(C)C(=O)OC1C(=CC23C1(C(C(=CC(C2=O)C4C(C4(C)C)CC3C)CO)O)O)C",
            name="Picato",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        30: similarity(
            smiles="CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
            name="Morphine",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        31: similarity(
            smiles="C1C2CC3CC1CC(C2)(C3)N",
            name="Amantadine",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        32: similarity(
            smiles="CC(C12CC3CC(C1)CC(C3)C2)N",
            name="Rimantadine",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        33: similarity(
            smiles="C1CC(N(C1)C(=O)CNC23CC4CC(C2)CC(C4)(C3)O)C#N",
            name="Vildagliptin",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        34: similarity(
            smiles="CC12CC3CC(C1)(CC(C3)(C2)N)C",
            name="Memantine",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        35: similarity(
            smiles="CN(C)CCOCC(=O)NC12CC3CC(C1)CC(C3)C2",
            name="Tromantadine",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        36: similarity(
            smiles="COC1=C(C=C(C=C1)C2=CC3=C(C=C2)C=C(C=C3)C(=O)O)C45CC6CC(C4)CC(C6)C5",
            name="Adapalene",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        37: similarity(
            smiles="C1C2CC2N(C1C#N)C(=O)C(C34CC5CC(C3)CC(C5)(C4)O)N",
            name="Saxagliptin",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        38: similarity(
            smiles="CC=C(C)C(=O)OC1C(=CC23C1(C(C(=CC(C2=O)C4C(C4(C)C)CC3C)CO)O)O)C",
            name="Picato",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        39: similarity(
            smiles="CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
            name="Morphine",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        40: similarity(
            smiles="C1C2CC3CC1CC(C2)(C3)N",
            name="Amantadine",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        41: similarity(
            smiles="CC(C12CC3CC(C1)CC(C3)C2)N",
            name="Rimantadine",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        42: similarity(
            smiles="C1CC(N(C1)C(=O)CNC23CC4CC(C2)CC(C4)(C3)O)C#N",
            name="Vildagliptin",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        43: similarity(
            smiles="CC12CC3CC(C1)(CC(C3)(C2)N)C",
            name="Memantine",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        44: similarity(
            smiles="CN(C)CCOCC(=O)NC12CC3CC(C1)CC(C3)C2",
            name="Tromantadine",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        45: similarity(
            smiles="COC1=C(C=C(C=C1)C2=CC3=C(C=C2)C=C(C=C3)C(=O)O)C45CC6CC(C4)CC(C6)C5",
            name="Adapalene",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        46: similarity(
            smiles="C1C2CC2N(C1C#N)C(=O)C(C34CC5CC(C3)CC(C5)(C4)O)N",
            name="Saxagliptin",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        47: similarity(
            smiles="CC1(C2CCC1(C(=O)C2)C)C",
            name="Camphor",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        48: similarity(
            smiles="CC1CCC(C(C1)O)C(C)C",
            name="Menthol",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        49: similarity(
            smiles="CN1CC(=O)N2C(C1=O)CC3=C(C2C4=CC5=C(C=C4)OCO5)NC6=CC=CC=C36",
            name="Tadalafil",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        50: similarity(
            smiles="CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
            name="Sildenafil",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        51: similarity(
            smiles="CC1(C2CCC1(C(=O)C2)C)C",
            name="Camphor",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        52: similarity(
            smiles="CC1CCC(C(C1)O)C(C)C",
            name="Menthol",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        
        53: similarity(
            smiles="CN1CC(=O)N2C(C1=O)CC3=C(C2C4=CC5=C(C=C4)OCO5)NC6=CC=CC=C36",
            name="Tadalafil",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        54: similarity(
            smiles="CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
            name="Sildenafil",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        55: AChE_score_1_task(),
        56: COX2_score_1_task(),
        57: PKCB_score_1_task(),
        58: FGFR1_score_1_task(),
        59: PTP1B_score_1_task(),
        60: AChE_score_2_task(),
        61: COX2_score_2_task(),
        62: PKCB_score_2_task(),
        63: FGFR1_score_2_task(),
        64: PTP1B_score_2_task(),
        65: AChE_score_3_task(),
        66: COX2_score_3_task(),
        67: PKCB_score_3_task(),
        68: FGFR1_score_3_task(),
        69: PTP1B_score_3_task(),
        70: AChE_score_4_task(),
        71: COX2_score_4_task(),
        72: PKCB_score_4_task(),
        73: FGFR1_score_4_task(),
        74: PTP1B_score_4_task(),
        75: MAOB_score_1_task(),
        76: MAOB_score_2_task(),
        77: MAOB_score_3_task(),
        78: MAOB_score_4_task(),
        79: AChE_MAOB_score_1_task(),
        80: AChE_MAOB_score_2_task(),
        81: AChE_MAOB_score_3_task(),
        82: AChE_MAOB_score_4_task(),
    }.get(benchmark_id)

    if benchmark_id in [
        3,
        4,
        5,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        26,
    ]:
        scoring_num_list = [1, 10, 100]
    elif benchmark_id in [6]:
        scoring_num_list = [159]
    elif benchmark_id in [7]:
        scoring_num_list = [250]
    elif benchmark_id in [25]:
        scoring_num_list = [100]
    elif benchmark_id in [0, 1, 2, 27, 28]:
        scoring_num_list = [1]
    else:
        scoring_num_list = [1, 10, 100]

    return benchmark, scoring_num_list  # type: ignore
