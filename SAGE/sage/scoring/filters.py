"""
Copyright (c) 2022 Hocheol Lim.
"""
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import Descriptors, rdMolDescriptors

import signal
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import rankdata

def handler(signum, frame):
    raise Exception()

def filter_ro5(mol: Mol) -> bool:
    
    weight = round(rdMolDescriptors._CalcMolWt(mol), 2)
    logp = Descriptors.MolLogP(mol)
    donor = rdMolDescriptors.CalcNumLipinskiHBD(mol)
    acceptor = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    
    aroms = rdMolDescriptors.CalcNumAromaticRings(mol)
    hatms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    catms = Descriptors.HeavyAtomCount(mol)-rdMolDescriptors.CalcNumHeteroatoms(mol)
    
    if weight > 500 or logp > 5 or donor > 5 or acceptor > 10:
        return False
    
    return True

def filter_ro3(mol: Mol) -> bool:
    
    weight = round(rdMolDescriptors._CalcMolWt(mol), 2)
    logp = Descriptors.MolLogP(mol)
    donor = rdMolDescriptors.CalcNumLipinskiHBD(mol)
    acceptor = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    
    if weight > 300 or logp > 3 or donor > 3 or acceptor > 3 or rotbonds > 3:
        return False
    
    return True


def filter_muegge(mol: Mol) -> bool:
    
    weight = round(rdMolDescriptors._CalcMolWt(mol), 2)
    logp = Descriptors.MolLogP(mol)
    donor = rdMolDescriptors.CalcNumLipinskiHBD(mol)
    acceptor = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    
    rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    aroms = rdMolDescriptors.CalcNumAromaticRings(mol)
    hatms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    catms = Descriptors.HeavyAtomCount(mol)-rdMolDescriptors.CalcNumHeteroatoms(mol)
    
    if weight > 600 or weight < 200 or logp > 6 or donor > 6 or acceptor > 12:
        return False

    if rotbonds > 15 or aroms > 7 or hatms < 2 or catms < 5:
        return False
    
    return True


def filter_cycle_len(mol: Mol, threshold=6) -> bool:
    import networkx as nx
    
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))

    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])

    if cycle_length > threshold:
        return False
    
    return True


def filter_rascore(mol: Mol, threshold=0.5) -> bool:
    from RAscore import RAscore_XGB

    compound = str(Chem.MolToSmiles(mol))
    xgb_scorer = RAscore_XGB.RAScorerXGB()
    score = xgb_scorer.predict(compound)
    
    if float(score) >= threshold:
        return True
    
    return False

def score_rascore(mol: Mol) -> float:
    from RAscore import RAscore_XGB
    
    compound = str(Chem.MolToSmiles(mol))
    xgb_scorer = RAscore_XGB.RAScorerXGB()
    score = xgb_scorer.predict(compound)
    return float(score)

def filter_solubility(mol: Mol, threshold=-6) -> bool:
    import soltrannet as stn
    
    try:
        compound = str(Chem.MolToSmiles(mol))
        score = list(stn.predict([compound], num_workers=1))[0][0]
        
        if score >= threshold:
            return True
    except Exception as exc:
        return False
    
    return False

def score_solubility(mol: Mol) -> float:
    import soltrannet as stn
    
    try:
        compound = str(Chem.MolToSmiles(mol))
        score = list(stn.predict([compound], num_workers=1))[0][0]
        return float(score)
    except Exception as exc:
        return float(-1000)
    
    return float(-1000)

def score_HIA(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    admet_model = {
        'caco2_wang_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'hia_hou_MACCS_PCFP_XGB': 'MACCS_PCFP',
        'pgp_broccatelli_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'bbb_martins_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'ppbr_az_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'cyp2d6_veith_MACCS_FCFP4_LGBM': 'MACCS_FCFP4',
        'cyp3a4_veith_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'cyp2c9_veith_MACCS_PCFP_LGBM': 'MACCS_PCFP',
        'ld50_zhu_MACCS_PCFP_RF': 'MACCS_PCFP',
        'herg_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'ames_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'dili_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
    }
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    
    temp_MACCS_ECFP6_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    temp_MACCS_FCFP4_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    temp_MACCS_PCFP_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    
    temp_name = 'hia_hou_MACCS_PCFP_XGB'
    
    clf = pickle.load(open('/home/admet/best_models/'+temp_name+'.pkl', 'rb'))
    fp_type = admet_model[temp_name]
    
    temp_fp = temp_MACCS_FCFP4_fp

    if fp_type == 'MACCS_FCFP4':
        temp_fp = temp_MACCS_FCFP4_fp
    elif fp_type == 'MACCS_ECFP6':
        temp_fp = temp_MACCS_ECFP6_fp
    elif fp_type == 'MACCS_PCFP':
        temp_fp = temp_MACCS_PCFP_fp
    
    temp_score = clf.predict_proba(temp_fp)[0][1]
    #print(temp_name,': ',temp_score)
    
    del clf
    
    return float(temp_score)

def score_BBB(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    admet_model = {
        'caco2_wang_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'hia_hou_MACCS_PCFP_XGB': 'MACCS_PCFP',
        'pgp_broccatelli_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'bbb_martins_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'ppbr_az_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'cyp2d6_veith_MACCS_FCFP4_LGBM': 'MACCS_FCFP4',
        'cyp3a4_veith_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'cyp2c9_veith_MACCS_PCFP_LGBM': 'MACCS_PCFP',
        'ld50_zhu_MACCS_PCFP_RF': 'MACCS_PCFP',
        'herg_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'ames_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'dili_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
    }
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    
    temp_MACCS_ECFP6_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    temp_MACCS_FCFP4_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    temp_MACCS_PCFP_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    
    temp_name = 'bbb_martins_MACCS_FCFP4_XGB'
    
    clf = pickle.load(open('/home/admet/best_models/'+temp_name+'.pkl', 'rb'))
    fp_type = admet_model[temp_name]
    
    temp_fp = temp_MACCS_FCFP4_fp

    if fp_type == 'MACCS_FCFP4':
        temp_fp = temp_MACCS_FCFP4_fp
    elif fp_type == 'MACCS_ECFP6':
        temp_fp = temp_MACCS_ECFP6_fp
    elif fp_type == 'MACCS_PCFP':
        temp_fp = temp_MACCS_PCFP_fp
    
    temp_score = clf.predict_proba(temp_fp)[0][1]
    #print(temp_name,': ',temp_score)
    
    del clf
    
    return float(temp_score)

def score_caco2(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    admet_model = {
        'caco2_wang_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'hia_hou_MACCS_PCFP_XGB': 'MACCS_PCFP',
        'pgp_broccatelli_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'bbb_martins_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'ppbr_az_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'cyp2d6_veith_MACCS_FCFP4_LGBM': 'MACCS_FCFP4',
        'cyp3a4_veith_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'cyp2c9_veith_MACCS_PCFP_LGBM': 'MACCS_PCFP',
        'ld50_zhu_MACCS_PCFP_RF': 'MACCS_PCFP',
        'herg_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'ames_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'dili_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
    }
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    
    temp_MACCS_ECFP6_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    temp_MACCS_FCFP4_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    temp_MACCS_PCFP_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    
    temp_name = 'caco2_wang_MACCS_FCFP4_RF'
    
    clf = pickle.load(open('/home/admet/best_models/'+temp_name+'.pkl', 'rb'))
    fp_type = admet_model[temp_name]
    
    temp_fp = temp_MACCS_FCFP4_fp

    if fp_type == 'MACCS_FCFP4':
        temp_fp = temp_MACCS_FCFP4_fp
    elif fp_type == 'MACCS_ECFP6':
        temp_fp = temp_MACCS_ECFP6_fp
    elif fp_type == 'MACCS_PCFP':
        temp_fp = temp_MACCS_PCFP_fp
    
    temp_score = clf.predict(temp_fp)[0]
    #print(temp_name,': ',temp_score)
    del clf
    
    return float(temp_score)

def filter_admet(mol: Mol, keyword='heuristic') -> bool:
    import sys
    import numpy as np
    from deepchem import deepchem
    import pickle
    
    flag_score = True
    
    maccskeys = deepchem.feat.MACCSKeysFingerprint()
    circular = deepchem.feat.CircularFingerprint()
    mol2vec = deepchem.feat.Mol2VecFingerprint()
    mordred = deepchem.feat.MordredDescriptors(ignore_3D=True)
    rdkit = deepchem.feat.RDKitDescriptors()
    pubchem = deepchem.feat.PubChemFingerprint()
    
    compound = str(Chem.MolToSmiles(mol))
    
    temp_maccskeys = maccskeys.featurize(compound)
    temp_circular = circular.featurize(compound)
    temp_mol2vec = mol2vec.featurize(compound)
    temp_mordred = mordred.featurize(compound)
    temp_rdkit = rdkit.featurize(compound)
    temp_pubchem = pubchem.featurize(compound)

    temp_fp = np.concatenate(
        (
        temp_maccskeys, temp_circular, temp_mol2vec,
        temp_rdkit, temp_mordred, temp_pubchem
        ), axis=1
    )

    temp_fp = np.nan_to_num(temp_fp, nan=0, posinf=0)
    
    admet_threshold = {
        'caco2_wang': -5.15,
        'bioavailability_ma': 0.5,
        'lipophilicity_astrazeneca': 10,
        'solubility_aqsoldb': -6,
        'hia_hou': 0.5,
        'pgp_broccatelli': 0.5,
        'bbb_martins': 0.5,
        'ppbr_az': 10,
        'vdss_lombardo': 10,
        'cyp2d6_veith': 0.5,
        'cyp3a4_veith': 0.5,
        'cyp2c9_veith': 0.5,
        'cyp2c9_substrate_carbonmangels': 0.5,
        'cyp2d6_substrate_carbonmangels': 0.5,
        'cyp3a4_substrate_carbonmangels': 0.5,
        'half_life_obach': 10,
        'clearance_microsome_az': 10,
        'clearance_hepatocyte_az': 10,
        'ld50_zhu': 10,
        'herg': 0.5,
        'ames': 0.5,
        'dili': 0.5,
    }
    
    # True: X should be lower, False: X should be higher
    admet_direction = {
        'caco2_wang': False,
        'bioavailability_ma': False,
        'lipophilicity_astrazeneca': False,
        'solubility_aqsoldb': False,
        'hia_hou': False,
        'pgp_broccatelli': True,
        'bbb_martins': False,
        'ppbr_az': True,
        'vdss_lombardo': True,
        'cyp2d6_veith': True,
        'cyp3a4_veith': True,
        'cyp2c9_veith': True,
        'cyp2c9_substrate_carbonmangels': True,
        'cyp2d6_substrate_carbonmangels': True,
        'cyp3a4_substrate_carbonmangels': True,
        'half_life_obach': False,
        'clearance_microsome_az': False,
        'clearance_hepatocyte_az': False,
        'ld50_zhu': True,
        'herg': True,
        'ames': True,
        'dili': True,
    }
    
    if keyword == 'all':
        name_list = ['caco2_wang', 'bioavailability_ma', 'lipophilicity_astrazeneca', 'solubility_aqsoldb', 'hia_hou', 'pgp_broccatelli', 'bbb_martins', 'ppbr_az', 'vdss_lombardo', 'cyp2d6_veith', 'cyp3a4_veith', 'cyp2c9_veith', 'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels', 'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az', 'ld50_zhu', 'herg', 'ames', 'dili']
        
        for temp_name in name_list:
            clf = pickle.load(open('/home/admet_xgb/models/'+temp_name+'_xgb.pkl', 'rb'))
            pred_score = clf.predict(temp_fp)[0]
            del clf
            
            if admet_direction[temp_name]:                
                if admet_threshold[temp_name] <= pred_score:
                    flag_score = False
                    return flag_score
            else:
                if pred_score <= admet_threshold[temp_name]:
                    flag_score = False
                    return flag_score            
    
    elif keyword == 'heuristic':
        name_list = ['hia_hou', 'pgp_broccatelli', 'ames', 'dili']
        
        for temp_name in name_list:
            clf = pickle.load(open('/home/admet_xgb/models/'+temp_name+'_xgb.pkl', 'rb'))
            pred_score = clf.predict(temp_fp)[0]
            del clf
            
            if admet_direction[temp_name]:                
                if admet_threshold[temp_name] <= pred_score:
                    flag_score = False
                    return flag_score
            else:
                if pred_score <= admet_threshold[temp_name]:
                    flag_score = False
                    return flag_score  

    else:
        clf = pickle.load(open('/home/admet_xgb/models/'+keyword+'_xgb.pkl', 'rb'))
        pred_score = clf.predict(temp_fp)[0]
        del clf

        if admet_direction[keyword]:                
            if admet_threshold[keyword] <= pred_score:
                flag_score = False
                return flag_score
        else:
            if pred_score <= admet_threshold[keyword]:
                flag_score = False
                return flag_score  
    
    return flag_score

def score_admet(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    admet_threshold = {
        'caco2_wang_MACCS_FCFP4_RF': -5.15,
        'hia_hou_MACCS_PCFP_XGB': 0.5,
        'pgp_broccatelli_MACCS_FCFP4_RF': 0.5,
        'bbb_martins_MACCS_FCFP4_XGB': 0.5,
        'ppbr_az_MACCS_FCFP4_RF': 80,
        'cyp2d6_veith_MACCS_FCFP4_LGBM': 0.55,
        'cyp3a4_veith_MACCS_FCFP4_XGB': 0.5,
        'cyp2c9_veith_MACCS_PCFP_LGBM': 0.5,
        'ld50_zhu_MACCS_PCFP_RF': 10,
        'herg_MACCS_FCFP4_RF': 0.5,
        'ames_MACCS_FCFP4_RF': 0.5,
        'dili_MACCS_FCFP4_XGB': 0.5,
    }
    
    # True: low, False: high
    admet_direction = {
        'caco2_wang_MACCS_FCFP4_RF': False,
        'hia_hou_MACCS_PCFP_XGB': False,
        'pgp_broccatelli_MACCS_FCFP4_RF': True,
        'bbb_martins_MACCS_FCFP4_XGB': False,
        'ppbr_az_MACCS_FCFP4_RF': True,
        'cyp2d6_veith_MACCS_FCFP4_LGBM': True,
        'cyp3a4_veith_MACCS_FCFP4_XGB': True,
        'cyp2c9_veith_MACCS_PCFP_LGBM': True,
        'ld50_zhu_MACCS_PCFP_RF': False,
        'herg_MACCS_FCFP4_RF': True,
        'ames_MACCS_FCFP4_RF': True,
        'dili_MACCS_FCFP4_XGB': True,
    }

    admet_model = {
        'caco2_wang_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'hia_hou_MACCS_PCFP_XGB': 'MACCS_PCFP',
        'pgp_broccatelli_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'bbb_martins_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'ppbr_az_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'cyp2d6_veith_MACCS_FCFP4_LGBM': 'MACCS_FCFP4',
        'cyp3a4_veith_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'cyp2c9_veith_MACCS_PCFP_LGBM': 'MACCS_PCFP',
        'ld50_zhu_MACCS_PCFP_RF': 'MACCS_PCFP',
        'herg_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'ames_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'dili_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
    }
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    
    temp_MACCS_ECFP6_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    temp_MACCS_FCFP4_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    temp_MACCS_PCFP_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)

    clf = pickle.load(open('/home/admet/best_models/herg_MACCS_FCFP4_RF.pkl', 'rb'))
    herg_fp = temp_MACCS_FCFP4_fp
    herg_score = clf.predict_proba(herg_fp)[0][1]
    #print('herg_MACCS_FCFP4_RF',': ',herg_score)
    
    if 0.75 <= float(herg_score):
        return float(0)

    name_binary_list = ['ames_MACCS_FCFP4_RF', 'cyp2c9_veith_MACCS_PCFP_LGBM', 'herg_MACCS_FCFP4_RF', 'cyp2d6_veith_MACCS_FCFP4_LGBM', 'cyp3a4_veith_MACCS_FCFP4_XGB', 'dili_MACCS_FCFP4_XGB', 'hia_hou_MACCS_PCFP_XGB', 'pgp_broccatelli_MACCS_FCFP4_RF']
    name_real_list = ['caco2_wang_MACCS_FCFP4_RF', 'ld50_zhu_MACCS_PCFP_RF', 'ppbr_az_MACCS_FCFP4_RF']
    
    score = 11.0

    for temp_name in name_binary_list:
        clf = pickle.load(open('/home/admet/best_models/'+temp_name+'.pkl', 'rb'))
        fp_type = admet_model[temp_name]
        
        temp_fp = temp_MACCS_FCFP4_fp

        if fp_type == 'MACCS_FCFP4':
            temp_fp = temp_MACCS_FCFP4_fp
        elif fp_type == 'MACCS_ECFP6':
            temp_fp = temp_MACCS_ECFP6_fp
        elif fp_type == 'MACCS_PCFP':
            temp_fp = temp_MACCS_PCFP_fp
        
        temp_score = clf.predict_proba(temp_fp)[0][1]
        #print(temp_name,': ',temp_score)

        del clf
        
        if admet_direction[temp_name]:                
            if admet_threshold[temp_name] <= float(temp_score):
                score = score - 1
                #print('minus')
        else:
            if float(temp_score) <= admet_threshold[temp_name]:
                score = score - 1
                #print('minus')
    
    for temp_name in name_real_list:
        clf = pickle.load(open('/home/admet/best_models/'+temp_name+'.pkl', 'rb'))
        fp_type = admet_model[temp_name]
        
        temp_fp = temp_MACCS_FCFP4_fp

        if fp_type == 'MACCS_FCFP4':
            temp_fp = temp_MACCS_FCFP4_fp
        elif fp_type == 'MACCS_ECFP6':
            temp_fp = temp_MACCS_ECFP6_fp
        elif fp_type == 'MACCS_PCFP':
            temp_fp = temp_MACCS_PCFP_fp
        
        temp_score = clf.predict(temp_fp)[0]
        #print(temp_name,': ',temp_score)

        del clf
        
        if admet_direction[temp_name]:
            if admet_threshold[temp_name] <= float(temp_score):
                score = score - 1
                #print('minus')
        else:
            if float(temp_score) <= admet_threshold[temp_name]:
                score = score - 1
                #print('minus')
    
    return round(float(score) / 11.0, 3)

def score_admet_wt_BBB(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    admet_threshold = {
        'caco2_wang_MACCS_FCFP4_RF': -5.15,
        'hia_hou_MACCS_PCFP_XGB': 0.5,
        'pgp_broccatelli_MACCS_FCFP4_RF': 0.5,
        'bbb_martins_MACCS_FCFP4_XGB': 0.5,
        'ppbr_az_MACCS_FCFP4_RF': 80,
        'cyp2d6_veith_MACCS_FCFP4_LGBM': 0.5,
        'cyp3a4_veith_MACCS_FCFP4_XGB': 0.5,
        'cyp2c9_veith_MACCS_PCFP_LGBM': 0.5,
        'ld50_zhu_MACCS_PCFP_RF': 10,
        'herg_MACCS_FCFP4_RF': 0.5,
        'ames_MACCS_FCFP4_RF': 0.5,
        'dili_MACCS_FCFP4_XGB': 0.5,
    }
    
    # True: low, False: high
    admet_direction = {
        'caco2_wang_MACCS_FCFP4_RF': False,
        'hia_hou_MACCS_PCFP_XGB': False,
        'pgp_broccatelli_MACCS_FCFP4_RF': True,
        'bbb_martins_MACCS_FCFP4_XGB': False,
        'ppbr_az_MACCS_FCFP4_RF': True,
        'cyp2d6_veith_MACCS_FCFP4_LGBM': True,
        'cyp3a4_veith_MACCS_FCFP4_XGB': True,
        'cyp2c9_veith_MACCS_PCFP_LGBM': True,
        'ld50_zhu_MACCS_PCFP_RF': False,
        'herg_MACCS_FCFP4_RF': True,
        'ames_MACCS_FCFP4_RF': True,
        'dili_MACCS_FCFP4_XGB': True,
    }

    admet_model = {
        'caco2_wang_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'hia_hou_MACCS_PCFP_XGB': 'MACCS_PCFP',
        'pgp_broccatelli_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'bbb_martins_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'ppbr_az_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'cyp2d6_veith_MACCS_FCFP4_LGBM': 'MACCS_FCFP4',
        'cyp3a4_veith_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'cyp2c9_veith_MACCS_PCFP_LGBM': 'MACCS_PCFP',
        'ld50_zhu_MACCS_PCFP_RF': 'MACCS_PCFP',
        'herg_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'ames_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'dili_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
    }
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    
    temp_MACCS_ECFP6_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    temp_MACCS_FCFP4_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    temp_MACCS_PCFP_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)

    clf = pickle.load(open('/home/admet/best_models/herg_MACCS_FCFP4_RF.pkl', 'rb'))
    herg_fp = temp_MACCS_FCFP4_fp
    herg_score = clf.predict_proba(herg_fp)[0][1]
    #print('herg_MACCS_FCFP4_RF',': ',herg_score)
    
    if 0.75 <= float(herg_score):
        return float(0)

    name_binary_list = ['ames_MACCS_FCFP4_RF', 'bbb_martins_MACCS_FCFP4_XGB', 'cyp2c9_veith_MACCS_PCFP_LGBM', 'cyp2d6_veith_MACCS_FCFP4_LGBM', 'cyp3a4_veith_MACCS_FCFP4_XGB', 'dili_MACCS_FCFP4_XGB', 'hia_hou_MACCS_PCFP_XGB', 'herg_MACCS_FCFP4_RF', 'pgp_broccatelli_MACCS_FCFP4_RF']
    name_real_list = ['caco2_wang_MACCS_FCFP4_RF', 'ld50_zhu_MACCS_PCFP_RF', 'ppbr_az_MACCS_FCFP4_RF']
    
    score = 12.0

    for temp_name in name_binary_list:
        clf = pickle.load(open('/home/admet/best_models/'+temp_name+'.pkl', 'rb'))
        fp_type = admet_model[temp_name]
        
        temp_fp = temp_MACCS_FCFP4_fp

        if fp_type == 'MACCS_FCFP4':
            temp_fp = temp_MACCS_FCFP4_fp
        elif fp_type == 'MACCS_ECFP6':
            temp_fp = temp_MACCS_ECFP6_fp
        elif fp_type == 'MACCS_PCFP':
            temp_fp = temp_MACCS_PCFP_fp
        
        temp_score = clf.predict_proba(temp_fp)[0][1]
        #print(temp_name,': ',temp_score)

        del clf
        
        if admet_direction[temp_name]:                
            if admet_threshold[temp_name] <= float(temp_score):
                score = score - 1
                #print('minus')
        else:
            if float(temp_score) <= admet_threshold[temp_name]:
                score = score - 1
                #print('minus')
    
    for temp_name in name_real_list:
        clf = pickle.load(open('/home/admet/best_models/'+temp_name+'.pkl', 'rb'))
        fp_type = admet_model[temp_name]
        
        temp_fp = temp_MACCS_FCFP4_fp

        if fp_type == 'MACCS_FCFP4':
            temp_fp = temp_MACCS_FCFP4_fp
        elif fp_type == 'MACCS_ECFP6':
            temp_fp = temp_MACCS_ECFP6_fp
        elif fp_type == 'MACCS_PCFP':
            temp_fp = temp_MACCS_PCFP_fp
        
        temp_score = clf.predict(temp_fp)[0]
        #print(temp_name,': ',temp_score)

        del clf
        
        if admet_direction[temp_name]:
            if admet_threshold[temp_name] <= float(temp_score):
                score = score - 1
                #print('minus')
        else:
            if float(temp_score) <= admet_threshold[temp_name]:
                score = score - 1
                #print('minus')
    
    return round(float(score) / 12.0, 3)

def score_admet_hia(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    admet_threshold = {
        'caco2_wang_MACCS_FCFP4_RF': -5.15,
        'hia_hou_MACCS_PCFP_XGB': 0.5,
        'pgp_broccatelli_MACCS_FCFP4_RF': 0.5,
        'bbb_martins_MACCS_FCFP4_XGB': 0.5,
        'ppbr_az_MACCS_FCFP4_RF': 80,
        'cyp2d6_veith_MACCS_FCFP4_LGBM': 0.5,
        'cyp3a4_veith_MACCS_FCFP4_XGB': 0.5,
        'cyp2c9_veith_MACCS_PCFP_LGBM': 0.5,
        'ld50_zhu_MACCS_PCFP_RF': 10,
        'herg_MACCS_FCFP4_RF': 0.5,
        'ames_MACCS_FCFP4_RF': 0.5,
        'dili_MACCS_FCFP4_XGB': 0.5,
    }
    
    # True: low, False: high
    admet_direction = {
        'caco2_wang_MACCS_FCFP4_RF': False,
        'hia_hou_MACCS_PCFP_XGB': False,
        'pgp_broccatelli_MACCS_FCFP4_RF': True,
        'bbb_martins_MACCS_FCFP4_XGB': False,
        'ppbr_az_MACCS_FCFP4_RF': True,
        'cyp2d6_veith_MACCS_FCFP4_LGBM': True,
        'cyp3a4_veith_MACCS_FCFP4_XGB': True,
        'cyp2c9_veith_MACCS_PCFP_LGBM': True,
        'ld50_zhu_MACCS_PCFP_RF': False,
        'herg_MACCS_FCFP4_RF': True,
        'ames_MACCS_FCFP4_RF': True,
        'dili_MACCS_FCFP4_XGB': True,
    }

    admet_model = {
        'caco2_wang_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'hia_hou_MACCS_PCFP_XGB': 'MACCS_PCFP',
        'pgp_broccatelli_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'bbb_martins_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'ppbr_az_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'cyp2d6_veith_MACCS_FCFP4_LGBM': 'MACCS_FCFP4',
        'cyp3a4_veith_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
        'cyp2c9_veith_MACCS_PCFP_LGBM': 'MACCS_PCFP',
        'ld50_zhu_MACCS_PCFP_RF': 'MACCS_PCFP',
        'herg_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'ames_MACCS_FCFP4_RF': 'MACCS_FCFP4',
        'dili_MACCS_FCFP4_XGB': 'MACCS_FCFP4',
    }
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    
    temp_MACCS_ECFP6_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    temp_MACCS_FCFP4_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    temp_MACCS_PCFP_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)

    clf = pickle.load(open('/home/admet/best_models/herg_MACCS_FCFP4_RF.pkl', 'rb'))
    herg_fp = temp_MACCS_FCFP4_fp
    herg_score = clf.predict_proba(herg_fp)[0][1]
    #print('herg_MACCS_FCFP4_RF',': ',herg_score)
    
    if 0.75 <= float(herg_score):
        return float(0)

    name_binary_list = ['hia_hou_MACCS_PCFP_XGB', 'herg_MACCS_FCFP4_RF']
    score = 2.0

    for temp_name in name_binary_list:
        clf = pickle.load(open('/home/admet/best_models/'+temp_name+'.pkl', 'rb'))
        fp_type = admet_model[temp_name]
        
        temp_fp = temp_MACCS_FCFP4_fp

        if fp_type == 'MACCS_FCFP4':
            temp_fp = temp_MACCS_FCFP4_fp
        elif fp_type == 'MACCS_ECFP6':
            temp_fp = temp_MACCS_ECFP6_fp
        elif fp_type == 'MACCS_PCFP':
            temp_fp = temp_MACCS_PCFP_fp
        
        temp_score = clf.predict_proba(temp_fp)[0][1]
        #print(temp_name,': ',temp_score)

        del clf
        
        if admet_direction[temp_name]:                
            if admet_threshold[temp_name] <= float(temp_score):
                score = score - 1
                #print('minus')
        else:
            if float(temp_score) <= admet_threshold[temp_name]:
                score = score - 1
                #print('minus')
    
    return round(float(score) / 2.0, 3)

class CustomRandomForestRegressor(RandomForestRegressor):
    
    def predict(self, X):
        
        y_pred = super().predict(X)
        
        ranks_up = rankdata(y_pred, method='min', axis=1)
        ranks_down = rankdata(-y_pred, method='min', axis=1)

        y_pred_rank = np.where(ranks_down <=1, 1.0,
                      np.where(ranks_down <=10, 0.9,
                      np.where(ranks_down <=50, 0.8,
                      np.where(ranks_down <=100, 0.7,
                      np.where(ranks_down <=200, 0.6,
                      np.where(ranks_up <= 1, 0.0,
                      np.where(ranks_up <= 10, 0.1, 
                      np.where(ranks_up <= 50, 0.2,
                      np.where(ranks_up <= 100, 0.3,
                      np.where(ranks_up <= 200, 0.4, 0.5))))))))))
        
        return y_pred_rank, y_pred

def score_gene_expression_MCF7(mol: Mol) -> float:
    import sys
    sys.path.append('/home/MFBERT')
    import numpy as np
    import pandas as pd
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestRegressor

    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    # Gene expression
    temp_cell_id = 0
    temp_cell_cond = [
        'MCF7_10um_24h',
        'VCAP_10um_24h',
        'PC3_10um_24h',
        'A375_10um_24h',
        'A549_10um_24h',
        'HA1E_10um_24h',
        'HEPG2_10um_24h',
        'HT29_10um_24h'
    ]
    
    temp_cell_model = [
    '/home/gene_expression/MCF7_E1500_D15_auto.pkl',
    '/home/gene_expression/VCAP_E3000_D15_auto.pkl',
    '/home/gene_expression/PC3_E2000_D20_sqrt.pkl',
    '/home/gene_expression/two_E2500_D20_auto.pkl',
    '/home/gene_expression/three_E1500_D15_sqrt.pkl',
    '/home/gene_expression/eight_E1500_D20_auto.pkl'
    ]
    
    clf_cell = pickle.load(open(temp_cell_model[0], 'rb'))    
    
    temp_cell_df = pd.read_csv('/home/gene_expression/condition_features.tsv', sep='\t', header=0)
    temp_cell_fp = temp_cell_df[temp_cell_df['condition'] == temp_cell_cond[0]].iloc[:, 1:].to_numpy().tolist()
    
    temp_gene_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT, temp_cell_fp)), axis=0)
    pred_gene = clf_cell.predict(temp_gene_fp)
    
    score = 1
    
    return float(score)