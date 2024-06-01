"""
Copyright (c) 2022 Hocheol Lim.
"""
from rdkit import Chem
from rdkit.Chem import Mol
from sage.scoring.filters import (
    filter_ro5,
    filter_ro3,
    filter_muegge,
    filter_cycle_len,
    filter_rascore,
    filter_solubility,
    filter_admet,
    score_rascore,
    score_admet,
)
def current_filters(mol: Mol) -> bool:
    flag = False
    
    flag_muegge = filter_muegge(mol)
    flag_rascore = filter_rascore(mol)
    flag_solubility = filter_solubility(mol)
    
    if flag_muegge == True and flag_rascore == True and flag_solubility == True:
        flag = True
    
    return flag

def filters_set_1(mol: Mol) -> bool:
    flag = False
    
    flag_muegge = filter_muegge(mol)
    
    if flag_muegge == True:
        flag = True
    
    return flag

def filters_set_2(mol: Mol) -> bool:
    flag = False
    
    flag_muegge = filter_muegge(mol)
    flag_solubility = filter_solubility(mol)
    
    if flag_muegge == True and flag_solubility == True:
        flag = True
    
    return flag

def solubility(mol: Mol) -> float:
    import soltrannet as stn
    
    compound = str(Chem.MolToSmiles(mol))
    score = list(stn.predict([compound]))[0][0]
    return float(score)

def dude_aces_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    temp_model = 'aces_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)

def dude_pgh2_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    temp_model = 'pgh2_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)

def dude_kpcb_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    temp_model = 'kpcb_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)

def dude_fgfr1_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    temp_model = 'fgfr1_MACCS_ECFP6_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)

def dude_ptn1_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    temp_model = 'ptn1_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)

def filtered_dude_aces_rascore_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1.0)
    
    temp_model = 'aces_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)

def filtered_dude_pgh2_rascore_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1.0)
    
    temp_model = 'pgh2_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)

def filtered_dude_kpcb_rascore_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier

    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1.0)

    temp_model = 'kpcb_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)

def filtered_dude_fgfr1_rascore_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1.0)
    
    temp_model = 'fgfr1_MACCS_ECFP6_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)

def filtered_dude_ptn1_rascore_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1.0)

    temp_model = 'ptn1_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)

def dude_aces_filter_1_score_1_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model = 'aces_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)


def dude_pgh2_filter_1_score_1_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model = 'pgh2_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)


def dude_kpcb_filter_1_score_1_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model = 'kpcb_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)


def dude_fgfr1_filter_1_score_1_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model = 'fgfr1_MACCS_ECFP6_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)


def dude_ptn1_filter_1_score_1_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model = 'ptn1_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)


def dude_aces_filter_1_score_2_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model = 'aces_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)


def dude_pgh2_filter_1_score_2_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model = 'pgh2_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)


def dude_kpcb_filter_1_score_2_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier

    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)

    temp_model = 'kpcb_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)


def dude_fgfr1_filter_1_score_2_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model = 'fgfr1_MACCS_ECFP6_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)


def dude_ptn1_filter_1_score_2_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)

    temp_model = 'ptn1_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)


def dude_aces_filter_1_score_3_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0
    
    temp_model = 'aces_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol) + score_sol
    
    del clf
    
    return float(score)


def dude_pgh2_filter_1_score_3_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)

    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0

    temp_model = 'pgh2_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol) + score_sol
    
    del clf
    
    return float(score)


def dude_kpcb_filter_1_score_3_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier

    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)

    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0

    temp_model = 'kpcb_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol) + score_sol
    
    del clf
    
    return float(score)


def dude_fgfr1_filter_1_score_3_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)

    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0
    
    temp_model = 'fgfr1_MACCS_ECFP6_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol) + score_sol
    
    del clf
    
    return float(score)


def dude_ptn1_filter_1_score_3_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0
    
    temp_model = 'ptn1_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol) + score_sol
    
    del clf
    
    return float(score)


def dude_aces_filter_1_score_4_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0
    
    temp_model = 'aces_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score_ra = score_rascore(mol)
    
    try:
        if float(score) >= 0.75 and score_sol >= 0.75 and score_ra >= 0.75:
            admet_score = score_admet(mol)
        else:
            admet_score = 0.0
    except:
        admet_score = 0.0
    
    score = float(score) + score_ra + score_sol + admet_score
    
    del clf
    
    return float(score)


def dude_pgh2_filter_1_score_4_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0
    
    temp_model = 'pgh2_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score_ra = score_rascore(mol)
    
    try:
        if float(score) >= 0.75 and score_sol >= 0.75 and score_ra >= 0.75:
            admet_score = score_admet(mol)
        else:
            admet_score = 0.0
    except:
        admet_score = 0.0
    
    score = float(score) + score_ra + score_sol + admet_score
    
    del clf
    
    return float(score)


def dude_kpcb_filter_1_score_4_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier

    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)

    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0
    
    temp_model = 'kpcb_MACCS_FCFP4_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_FCFP4)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score_ra = score_rascore(mol)
    
    try:
        if float(score) >= 0.75 and score_sol >= 0.75 and score_ra >= 0.75:
            admet_score = score_admet(mol)
        else:
            admet_score = 0.0
    except:
        admet_score = 0.0
    
    score = float(score) + score_ra + score_sol + admet_score
    
    del clf
    
    return float(score)


def dude_fgfr1_filter_1_score_4_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0
    
    temp_model = 'fgfr1_MACCS_ECFP6_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score_ra = score_rascore(mol)
    
    try:
        if float(score) >= 0.75 and score_sol >= 0.75 and score_ra >= 0.75:
            admet_score = score_admet(mol)
        else:
            admet_score = 0.0
    except:
        admet_score = 0.0
    
    score = float(score) + score_ra + score_sol + admet_score
    
    del clf
    
    return float(score)


def dude_ptn1_filter_1_score_4_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import time
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0
    
    temp_model = 'ptn1_MACCS_PCFP_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    #temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    
    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score_ra = score_rascore(mol)
    
    try:
        if float(score) >= 0.75 and score_sol >= 0.75 and score_ra >= 0.75:
            admet_score = score_admet(mol)
        else:
            admet_score = 0.0
    except:
        admet_score = 0.0
    
    score = float(score) + score_ra + score_sol + admet_score
    
    del clf
    
    return float(score)

def dude_aofb_filter_1_score_1_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model = 'aofb_MACCS_ECFP6_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    del clf
    
    return float(score)

def dude_aofb_filter_1_score_2_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model = 'aofb_MACCS_ECFP6_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol)
    
    del clf
    
    return float(score)

def dude_aofb_filter_1_score_3_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier

    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)

    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0

    temp_model = 'aofb_MACCS_ECFP6_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score = float(score) + score_rascore(mol) + score_sol
    
    del clf
    
    return float(score)

def dude_aofb_filter_1_score_4_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import time
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0
    
    temp_model = 'aofb_MACCS_ECFP6_RF'
    clf = pickle.load(open('/home/dude/best_models/'+temp_model+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    #temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    
    temp_fp = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    score = clf.predict_proba(temp_fp)[0][1]
    
    score_ra = score_rascore(mol)
    
    try:
        if float(score) >= 0.75 and score_sol >= 0.75 and score_ra >= 0.75:
            admet_score = score_admet(mol)
        else:
            admet_score = 0.0
    except:
        admet_score = 0.0
    
    score = float(score) + score_ra + score_sol + admet_score
    
    del clf
    
    return float(score)

def dude_aces_aofb_filter_1_score_1_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model_aces = 'aces_MACCS_PCFP_RF'
    temp_model_aofb = 'aofb_MACCS_ECFP6_RF'
    
    clf_aces = pickle.load(open('/home/dude/best_models/'+temp_model_aces+'.pkl', 'rb'))
    clf_aofb = pickle.load(open('/home/dude/best_models/'+temp_model_aofb+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp_aces = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    temp_fp_aofb = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    
    score_aces = clf_aces.predict_proba(temp_fp_aces)[0][1]
    score_aofb = clf_aofb.predict_proba(temp_fp_aofb)[0][1]
    
    score = float(score_aces) * 0.5 + float(score_aofb) * 0.5
    
    del clf_aces
    del clf_aofb
    
    return float(score)

def dude_aces_aofb_filter_1_score_2_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    temp_model_aces = 'aces_MACCS_PCFP_RF'
    temp_model_aofb = 'aofb_MACCS_ECFP6_RF'
    
    clf_aces = pickle.load(open('/home/dude/best_models/'+temp_model_aces+'.pkl', 'rb'))
    clf_aofb = pickle.load(open('/home/dude/best_models/'+temp_model_aofb+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp_aces = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    temp_fp_aofb = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    
    score_aces = clf_aces.predict_proba(temp_fp_aces)[0][1]
    score_aofb = clf_aofb.predict_proba(temp_fp_aofb)[0][1]
    
    score = float(score_aces) * 0.5 + float(score_aofb) * 0.5
    
    del clf_aces
    del clf_aofb
    
    score = float(score) + score_rascore(mol)
    
    return float(score)

def dude_aces_aofb_filter_1_score_3_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import time
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier

    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)

    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0

    temp_model_aces = 'aces_MACCS_PCFP_RF'
    temp_model_aofb = 'aofb_MACCS_ECFP6_RF'
    
    clf_aces = pickle.load(open('/home/dude/best_models/'+temp_model_aces+'.pkl', 'rb'))
    clf_aofb = pickle.load(open('/home/dude/best_models/'+temp_model_aofb+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp_aces = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    temp_fp_aofb = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    
    score_aces = clf_aces.predict_proba(temp_fp_aces)[0][1]
    score_aofb = clf_aofb.predict_proba(temp_fp_aofb)[0][1]
    
    score = float(score_aces) * 0.5 + float(score_aofb) * 0.5
    
    del clf_aces
    del clf_aofb
    
    score = float(score) + score_rascore(mol) + score_sol
    
    return float(score)

def dude_aces_aofb_filter_1_score_4_score(mol: Mol) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import time
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    
    flag = filters_set_1(mol)
    if(flag == False):
        return float(-1000)
    
    if score_solubility(mol) < -6.0:
        score_sol = 0.0
    else:
        score_sol = 1.0
    
    temp_model_aces = 'aces_MACCS_PCFP_RF'
    temp_model_aofb = 'aofb_MACCS_ECFP6_RF'
    
    clf_aces = pickle.load(open('/home/dude/best_models/'+temp_model_aces+'.pkl', 'rb'))
    clf_aofb = pickle.load(open('/home/dude/best_models/'+temp_model_aofb+'.pkl', 'rb'))
    
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    #temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()

    temp_fp_aces = np.expand_dims(np.concatenate((temp_maccs_keys, temp_PCFP)), axis=0)
    temp_fp_aofb = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6)), axis=0)
    
    score_aces = clf_aces.predict_proba(temp_fp_aces)[0][1]
    score_aofb = clf_aofb.predict_proba(temp_fp_aofb)[0][1]
    
    score = float(score_aces) * 0.5 + float(score_aofb) * 0.5
    
    del clf_aces
    del clf_aofb
    
    score_ra = score_rascore(mol)
    
    try:
        if float(score) >= 0.75 and score_sol >= 0.75 and score_ra >= 0.75:
            admet_score = score_admet_wt_BBB(mol)
        else:
            admet_score = 0.0
    except:
        admet_score = 0.0
    
    score = float(score) + score_ra + score_sol + admet_score
    
    return float(score)

