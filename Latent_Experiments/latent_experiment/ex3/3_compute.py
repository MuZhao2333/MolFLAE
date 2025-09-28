from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sdf_file', type=str,default='./latent_experiment/ex3/output')
parser.add_argument('--switch', type=str,help='zh or zx')
args= parser.parse_args()
if (args.switch == 'zh'):
    file = args.sdf_file+'/switchZh.sdf'
elif (args.switch == 'zx'):
    file = args.sdf_file+'/switchZx.sdf'
else:
    print('please input zh or zx')
    exit()
# Load molecules from SDF (keep None to preserve indexing)
supplier = Chem.SDMolSupplier(file, removeHs=True,sanitize=True)
mols = list(supplier)

topo_sims = []
shape_sims = []

# Compare molecules at indices 3k and 3k+2
for k in range(len(mols) // 3):
    idx1 = 3 * k
    idx2 = 3 * k + 2
    mol1 = mols[idx1]
    mol2 = mols[idx2]

    if mol1 is None or mol2 is None:
        print(f"Skipping pair ({idx1}, {idx2}): one or both molecules are None.")
        continue

    # fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    # fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

    fp1 = MACCSkeys.GenMACCSKeys(mol1)
    fp2 = MACCSkeys.GenMACCSKeys(mol2)

    topo_sim = DataStructs.TanimotoSimilarity(fp1, fp2)

    # Compute shape similarity using existing coordinates
    shape_dist = ShapeTanimotoDist(mol1, mol2)
    shape_sim = 1 - shape_dist

    
    topo_sims.append(topo_sim)
    shape_sims.append(shape_sim)


# Convert to NumPy arrays and filter out NaNs
topo_array = np.array(topo_sims)
topo_array_clean = topo_array[~np.isnan(topo_array)]
shape_array = np.array(shape_sims)
shape_array_clean = shape_array[~np.isnan(shape_array)]

# Compute mean and SEM
def mean_sem(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    sem = np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else float('nan')
    return mean,std, sem

topo_mean,topo_std, topo_sem = mean_sem(topo_array_clean)
shape_mean, shape_std,shape_sem = mean_sem(shape_array_clean)

print("\nSummary:")
print(f"file: {file}")
print(f"Topology Similarity: mean = {topo_mean:.3f}, STD = {topo_std:.3f}, SEM = {topo_sem:.3f}")
print(f"Shape Similarity:         mean = {shape_mean:.3f}, STD = {shape_std:.3f}, SEM = {shape_sem:.3f}")
