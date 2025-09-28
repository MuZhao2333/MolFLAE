import sys
import os
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED,MACCSkeys,DataStructs
from scipy.stats import pearsonr
import sys
import os
import argparse
from rdkit import Chem
parser = argparse.ArgumentParser()
parser.add_argument('--sdf_file', type=str,default='./latent_experiment/ex2/generated/merged.sdf')
parser.add_argument('--n_interpolations', type=int, default=12)
args = parser.parse_args()
sdf_file = args.sdf_file
supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
mols = list(supplier)
N = args.n_interpolations
M = len(mols) // N  
features = {
    'Labute ASA': Descriptors.LabuteASA,
    "TPSA": Descriptors.TPSA,
    "logP": Crippen.MolLogP,
    "MR": Crippen.MolMR,
    "sp3frac": rdMolDescriptors.CalcFractionCSP3,
    "BertzCT": Descriptors.BertzCT,
    "QED": QED.qed,
    "Similarity": None,
}
feat_arr = {name: np.full((M, N), np.nan) for name in features.keys()}

# for i, mol in enumerate(mols):
#     m = i // N
#     n = i % N
#     if mol is None:
#         continue
#     for name, func in features.items():
#         try:
#             feat_arr[name][m, n] = func(mol)
#         except Exception as e:
#             pass
for i in range(M):
    start_mol = mols[i * N]
    end_mol = mols[(i + 1) * N - 1]
    start_fp = MACCSkeys.GenMACCSKeys(start_mol)
    end_fp = MACCSkeys.GenMACCSKeys(end_mol)
    features['Similarity'] = lambda mol: (DataStructs.TanimotoSimilarity(end_fp, MACCSkeys.GenMACCSKeys(mol)) - DataStructs.TanimotoSimilarity(start_fp, MACCSkeys.GenMACCSKeys(mol)))/(DataStructs.TanimotoSimilarity(end_fp, MACCSkeys.GenMACCSKeys(mol)) + DataStructs.TanimotoSimilarity(start_fp, MACCSkeys.GenMACCSKeys(mol)))
    for j in range(N):
        mol = mols[i * N + j]
        if mol is None:
            continue
        for name, func in features.items():
            try:
                feat_arr[name][i, j] = func(mol)
            except Exception as e:
                pass
def rowwise_pearson(matrix,correct_mono=True):
    x = np.arange(matrix.shape[1])
    r_list, p_list = [], []
    for row in matrix:
        if np.isnan(row).all():
            r_list.append(np.nan)
            p_list.append(np.nan)
            continue
        try:
            valid = ~np.isnan(row)
            r, p = pearsonr(x[valid], row[valid])
            if correct_mono:
                r = np.sign(row[-1]-row[0]) * r
            p = -np.log10(p)
        except:
            r, p = np.nan, np.nan
        r_list.append(r)
        p_list.append(p)
    return np.nanmean(r_list), np.nanmean(p_list)

print(f"\nSummary@{N}:")
for name, arr in feat_arr.items():
    if name == 'Similarity':

        r, p = rowwise_pearson(arr,False)
        print(f"{name} mean: r = {r:.4f}, -log10(p) = {p:.4f}")
    else:
        r, p = rowwise_pearson(arr,True)
        print(f"{name} mean: r = {r:.4f}, -log10(p) = {p:.4f}")   