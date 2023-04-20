"""
Mutation methods and functions

Copyright (c) 2022 Hocheol Lim.
"""

import random
from typing import Optional

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from sage.utils.molecules import mol_is_ok, ring_is_ok

rdBase.DisableLog("rdApp.error")


def delete_atom() -> str:
    choices = [
        "[*:1]~[D1:2]>>[*:1]",
        "[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]",
        "[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]",
        "[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]",
        "[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]",
    ]
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]

    return np.random.choice(choices, p=p)


def append_atom() -> str:
    choices = [
        ["single", ["C", "N", "O", "F", "S", "Cl", "Br"], 7 * [1.0 / 7.0]],
        ["double", ["C", "N", "O"], 3 * [1.0 / 3.0]],
        ["triple", ["C", "N"], 2 * [1.0 / 2.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == "single":
        rxn_smarts = "[*;!H0:1]>>[*:1]X".replace("X", "-" + new_atom)
    if BO == "double":
        rxn_smarts = "[*;!H0;!H1:1]>>[*:1]X".replace("X", "=" + new_atom)
    if BO == "triple":
        rxn_smarts = "[*;H3:1]>>[*:1]X".replace("X", "#" + new_atom)

    return rxn_smarts


def insert_atom() -> str:
    choices = [
        ["single", ["C", "N", "O", "S"], 4 * [1.0 / 4.0]],
        ["double", ["C", "N"], 2 * [1.0 / 2.0]],
        ["triple", ["C"], [1.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == "single":
        rxn_smarts = "[*:1]~[*:2]>>[*:1]X[*:2]".replace("X", new_atom)
    if BO == "double":
        rxn_smarts = "[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]".replace("X", new_atom)
    if BO == "triple":
        rxn_smarts = "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]".replace("X", new_atom)

    return rxn_smarts


def change_bond_order() -> str:
    choices = [
        "[*:1]!-[*:2]>>[*:1]-[*:2]",
        "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
        "[*:1]#[*:2]>>[*:1]=[*:2]",
        "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
    ]
    p = [0.45, 0.45, 0.05, 0.05]

    return np.random.choice(choices, p=p)


def delete_cyclic_bond() -> str:
    return "[*:1]@[*:2]>>([*:1].[*:2])"


def add_ring() -> str:
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    p = [0.05, 0.05, 0.45, 0.45]

    return np.random.choice(choices, p=p)


def change_atom(mol: Chem.Mol) -> str:
    choices = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
    p = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

    X = np.random.choice(choices, p=p)
    while not mol.HasSubstructMatch(Chem.MolFromSmarts("[" + X + "]")):
        X = np.random.choice(choices, p=p)
    Y = np.random.choice(choices, p=p)
    while Y == X:
        Y = np.random.choice(choices, p=p)

    return "[X:1]>>[Y:1]".replace("X", X).replace("Y", Y)

def change_bridged_rings(mol: Chem.Mol) -> Optional[Chem.Mol]:
    
    ring_smarts = Chem.MolFromSmarts("[A;R:1]~[*;R:2]~[*;R:3]~[A;R:4]")
    if not mol.HasSubstructMatch(ring_smarts):
        return None
    
    rxn_1 = Chem.AllChem.ReactionFromSmarts("[A;R:1]~[*;R:2]~[*;R:3]~[A;R:4]>>[*:1](~[*:2]~[*:3]1)-[#6]-[*:4]~1")
    rxn_2 = Chem.AllChem.ReactionFromSmarts("[A;R:1]~[*;R:2]~[*;R:3]~[A;R:4]>>[*:1](~[*:2]~[*:3]1)-[#6]-[#6]-[*:4]~1")
    rxn_3 = Chem.AllChem.ReactionFromSmarts("[A;R:1]~[*;R:2]~[*;R:3]~[A;R:4]>>[*:1](~[*:2]~[*:3]1)-[#6]-[#6]-[#6]-[*:4]~1")
    
    products_1 = list(rxn_1.RunReactants((mol,)))
    products_2 = list(rxn_2.RunReactants((mol,)))
    products_3 = list(rxn_3.RunReactants((mol,)))
    
    new_mols = []
    for temp_mol in products_1:
        if mol_is_ok(temp_mol[0]):
            new_mols.append(Chem.MolToSmiles(temp_mol[0]))
    for temp_mol in products_2:
        if mol_is_ok(temp_mol[0]):
            new_mols.append(Chem.MolToSmiles(temp_mol[0]))
    for temp_mol in products_3:
        if mol_is_ok(temp_mol[0]):
            new_mols.append(Chem.MolToSmiles(temp_mol[0]))
            
    new_mols = list(dict.fromkeys(new_mols))
    return [Chem.MolFromSmiles(temp) for temp in new_mols]

def virtual_synthesis(mol: Chem.Mol, timeout='1m', retry=3, num=10) -> Optional[Chem.Mol]:
    import os
    import uuid
    import subprocess
    import pandas as pd
    
    DINGOS_ROOT = '/home/DINGOS'
    
    init = 0
    flag_error = True
    
    while init < retry and flag_error:
        init = init+1
        flag_error=False
        try:
            filename = str(uuid.uuid4()).replace("-","")
            filename_input = filename + '_dingos_input.csv'
            filename_log = filename + '_dingos.log'
            filename_output = filename + '_dingos_output.tsv'
            
            f1 = open(filename_input, 'w')
            f1.write('Data sets ,,,,,,,,,,,,,\n')
            f1.write('Building block label,Reaction label,Descriptor type,Machine learning method ,"Building block mass limit (lower,upper)",Preselected reactions,,,,,,,,\n')
            f1.write('TEST_database,TEST_database,MACCSkeys,MLP_method,"0,400",None,,,,,,,,\n')
            f1.write('Template info ,,,,Run limits,,,,,Filter parameters,,,,Machine learning parameters\n')
            f1.write('Author name,Template name,Template SMILES,,Product mass limit ,Product limit ,Start fragment limit ,Reaction step limit,,Flagged reactions,Included subgroups in the starting molecule,Excluded subgroups,,Use Machine learning method for start molecule selection\n')
            f1.write('sage,DINGOS,')
            f1.write(Chem.MolToSmiles(mol))
            f1.write(',,600,'+str(num)+',20,2,,None,None,None,,FALSE\n')
            f1.close()
            
            cmd_timeout = 'timeout '+timeout+' '
            cmd_1_1 = 'sh '
            cmd_1_2 = '/run_dingos.sh ' + os.getcwd() + '/' + filename
            cmd_1 = cmd_timeout + cmd_1_1 + DINGOS_ROOT + cmd_1_2
            cmd_1_flag = subprocess.call([cmd_1], shell=True)
            
            output = pd.read_csv(filename_output, sep='\t', header=0)
            new_mol_trial = output.iloc[:, 1]
            
            new_mols = []
            for m in new_mol_trial:
                m = Chem.MolFromSmiles(m)
                if mol_is_ok(m) and ring_is_ok(m):
                    new_mols.append(m)
            
            os.system('rm -rf '+filename+'*')
            
            if len(new_mols) > 0:
                return new_mols
            else:
                flag_error=True
            
        except:
            flag_error=True
            os.system('rm -rf '+filename+'*')

def mutate(
    mol: Chem.Mol, mutation_rate: float, num_trials: int = 10
) -> Optional[Chem.Mol]:
    if random.random() > mutation_rate:
        return mol

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        return mol
    
    p = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for _ in range(num_trials):
        proba = random.random()
        if proba <= 0.25:
            rxn_smarts_list = 7 * [""]
            rxn_smarts_list[0] = insert_atom()
            rxn_smarts_list[1] = change_bond_order()
            rxn_smarts_list[2] = delete_cyclic_bond()
            rxn_smarts_list[3] = add_ring()
            rxn_smarts_list[4] = delete_atom()
            rxn_smarts_list[5] = change_atom(mol)
            rxn_smarts_list[6] = append_atom()
            rxn_smarts = np.random.choice(rxn_smarts_list, p=p)
    
            rxn = Chem.AllChem.ReactionFromSmarts(rxn_smarts)
    
            new_mol_trial = rxn.RunReactants((mol,))
    
            new_mols = []
            for m in new_mol_trial:
                m = m[0]
                if mol_is_ok(m) and ring_is_ok(m):
                    new_mols.append(m)
    
            if new_mols is not None and len(new_mols) > 0:
                return random.choice(new_mols)
        elif proba <= 0.95:
            new_mols = []
            new_mols = virtual_synthesis(mol)
            if new_mols is not None and len(new_mols) > 0:
                return random.choice(new_mols)
        else:
            new_mols = []
            new_mols = change_bridged_rings(mol)
            if new_mols is not None and len(new_mols) > 0:
                return random.choice(new_mols)

    return None
