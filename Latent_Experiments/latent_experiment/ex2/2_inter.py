import sys
import os
import argparse  
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
import numpy as np
from rdkit import Chem
from model.train_loop import TrainLoop, center_pos
from config.config import load_config
import torch
import pytorch_lightning as pl
import random
import torch.nn.functional as F
from tqdm import tqdm
from utils.build_mol import MoleculeBuilder

# Atom type mapping dictionary
MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    6: 0,
    7: 1,
    8: 2,
    9: 3,
    15: 4,
    16: 5,
    17: 6,
    35: 7,
    53: 8,
}
MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}

def get_valid_molecules(root_folder):
    valid_molecules = []
    total_molecules = 0
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.sdf'):
                sdf_file_path = os.path.join(foldername, filename)
                suppl = Chem.SDMolSupplier(sdf_file_path)
                total_molecules += len(suppl)

    progress_bar = tqdm(total=total_molecules, desc='Processing molecules', unit='mol')

    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.sdf'):
                sdf_file_path = os.path.join(foldername, filename)
                suppl = Chem.SDMolSupplier(sdf_file_path)
                for mol in suppl:
                    if mol is not None:
                        atoms = mol.GetAtoms()
                        atom_types = [a.GetAtomicNum() for a in atoms]
                        # Check if all atom types in the molecule are in MAP_ATOM_TYPE_ONLY_TO_INDEX
                        valid_molecule = all(atomic_num in MAP_ATOM_TYPE_ONLY_TO_INDEX for atomic_num in atom_types)

                        if valid_molecule:
                            valid_molecules.append(mol)
                        progress_bar.update(1)

    progress_bar.close()
    return valid_molecules

def save_molecule_pairs(output_folder, start_mol, end_mol, index):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start_output_path = os.path.join(output_folder, f'{index}_start.sdf')
    start_writer = Chem.SDWriter(start_output_path)
    start_writer.write(start_mol)
    start_writer.close()

    end_output_path = os.path.join(output_folder, f'{index}_end.sdf')
    end_writer = Chem.SDWriter(end_output_path)
    end_writer.write(end_mol)
    end_writer.close()

def generate_and_save_molecule_pairs(root_folder, output_folder, num_pairs):
    valid_molecules = get_valid_molecules(root_folder)
    num_valid_molecules = len(valid_molecules)
    print("Valid molecules:", num_valid_molecules)

    if num_valid_molecules < 2:
        print("Insufficient eligible molecules to generate molecule pairs.")
        return

    molecule_indices = list(range(num_valid_molecules))
    generated_pairs = set()  # To store generated molecule pairs and avoid duplicates

    for i in range(num_pairs):
        # Randomly select two different molecule indices
        while True:
            index1 = random.randint(0, num_valid_molecules - 1)
            index2 = random.randint(0, num_valid_molecules - 1)
            if index1 != index2:
                # Place the smaller index first to ensure each pair is only generated once
                min_index, max_index = sorted([index1, index2])
                pair_key = (min_index, max_index)
                if pair_key not in generated_pairs:
                    generated_pairs.add(pair_key)
                    save_molecule_pairs(output_folder, valid_molecules[index1], valid_molecules[index2], i + 1)
                    break

    print(f"Successfully generated and saved {len(generated_pairs)} pairs of molecules to the {output_folder} folder.")

def load_molecule_pairs(folder_path):
    """
    Load molecule pair files and store them in a list. Each list element is a dictionary containing information of two molecules.

    Args:
        folder_path (str): Path to the folder containing molecule pair files.

    Returns:
        list: A list containing molecule pair information, where each element is a dictionary with keys 'mol_start' and 'mol_end'.
    """
    molecule_pairs = []
    start_files = sorted([f for f in os.listdir(folder_path) if f.endswith('_start.sdf')])
    
    for start_file in start_files:
        index = start_file.split('_')[0]
        end_file = f"{index}_end.sdf"
        if not os.path.exists(os.path.join(folder_path, end_file)):
            print(f"Corresponding end file {end_file} does not exist, skipping molecule pair {start_file}")
            continue
        start_mol = Chem.SDMolSupplier(os.path.join(folder_path, start_file))[0]
        if start_mol is None:
            print(f"Failed to read start molecule {start_file}, skipping")
            continue
        end_mol = Chem.SDMolSupplier(os.path.join(folder_path, end_file))[0]
        if end_mol is None:
            print(f"Failed to read end molecule {end_file}, skipping")
            continue
        start_atoms = start_mol.GetAtoms()
        start_atom_types = [a.GetAtomicNum() for a in start_atoms]
        start_conformer = start_mol.GetConformer()
        start_positions = start_conformer.GetPositions()
        start_atom_indices = start_atom_types

        end_atoms = end_mol.GetAtoms()
        end_atom_types = [a.GetAtomicNum() for a in end_atoms]
        end_conformer = end_mol.GetConformer()
        end_positions = end_conformer.GetPositions()
        end_atom_indices = end_atom_types

        molecule_pairs.append({
            'mol_start': {
                'x': torch.tensor(start_positions, dtype=torch.float32),
                'h': torch.tensor(start_atom_indices, dtype=torch.long),
                'atom_num': len(start_atom_indices)
            },
            'mol_end': {
                'x': torch.tensor(end_positions, dtype=torch.float32),
                'h': torch.tensor(end_atom_indices, dtype=torch.long),
                'atom_num': len(end_atom_indices)
            }
        })
    return molecule_pairs

def interpolate_and_generate(model, mol_start_data, mol_end_data, n_interpolations):
    h_start, x_start = mol_start_data['h'], mol_start_data['x']
    h_end, x_end = mol_end_data['h'], mol_end_data['x']
    K = model.cfg['encoder_config']['ligand_v_dim']
    device = model.device
    x_start = x_start.to(device)
    x_end = x_end.to(device)
    h_start = torch.tensor([MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h_start]).to(device)
    h_end = torch.tensor([MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h_end]).to(device)

    one_hot_h_start = F.one_hot(h_start, K).float().to(device)
    one_hot_h_end = F.one_hot(h_end, K).float().to(device)

    batch_ligan_orig_start = torch.zeros_like(h_start).to(device)
    batch_ligan_orig_end = torch.zeros_like(h_end).to(device)
    x_start, _ = center_pos(x_start, batch_ligan_orig_start, mode=True)
    x_end, _ = center_pos(x_end, batch_ligan_orig_end, mode=True)

    with torch.no_grad():
        Zh_start, Zx_start, _, _, _ = model.encode(one_hot_h_start, x_start, batch_ligan_orig_start, deterministic=True)
        Zh_end, Zx_end, _, _, _ = model.encode(one_hot_h_end, x_end, batch_ligan_orig_end, deterministic=True)

    alpha_values = torch.linspace(0, 1, n_interpolations)
    Zh_interp = torch.stack([Zh_start + alpha * (Zh_end - Zh_start) for alpha in alpha_values])
    Zx_interp = torch.stack([Zx_start + alpha * (Zx_end - Zx_start) for alpha in alpha_values])

    atom_num_start = mol_start_data['atom_num']
    atom_num_end = mol_end_data['atom_num']
    atom_num_interp = [round(int(atom_num_start) + float(alpha) * (int(atom_num_end) - int(atom_num_start))) for alpha in alpha_values]
    atom_num_interp = [max(num, 1) for num in atom_num_interp] 
    interpolation_results = []
    for i in range(n_interpolations):
        interpolation_results.append({
            'Zh': Zh_interp[i],
            'Zx': Zx_interp[i],
            'atom_num': atom_num_interp[i]
        })
    return interpolation_results

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False 
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' 
    torch.use_deterministic_algorithms(True)

def build_batch_and_sample(model, all_interpolation_results, group_size, save_dir):
    print(len(all_interpolation_results))
    group_num = max(len(all_interpolation_results) // group_size, 1)
    interpolate_num = len(all_interpolation_results[0])
    batches = []
    global_nodes = 10
    output_mols = []
    for i in range(group_num):
        start_idx = group_size * i
        batch_ligand = []
        batch_h = []
        batch_x = []
        global_batch = []
        batch_idx = 0
        for k in range(start_idx, min(start_idx + group_size, len(all_interpolation_results))):
            interpolate = all_interpolation_results[k]
            for j in range(interpolate_num):
                Zh = interpolate[j]['Zh']
                Zx = interpolate[j]['Zx']
                atom_num = interpolate[j]['atom_num']
                atom_index_each = torch.ones(atom_num, dtype=torch.long) * batch_idx
                global_node_each = torch.ones(global_nodes, dtype=torch.long) * batch_idx
                batch_idx += 1
                batch_h.append(Zh)
                batch_x.append(Zx)
                batch_ligand.append(atom_index_each)
                global_batch.append(global_node_each)
        device = model.device
        batch_h = (torch.cat(batch_h, dim=0)).to(device)
        batch_x = (torch.cat(batch_x, dim=0)).to(device)
        batch_ligand = (torch.cat(batch_ligand, dim=0)).to(device)
        global_batch = (torch.cat(global_batch, dim=0)).to(device)
        bat = {'batch_h': batch_h, 'batch_x': batch_x, 'batch_ligand': batch_ligand}
        batches.append(bat)
        with torch.no_grad():
            theta_chain, sample_chain, y_chain = model.decoder.sample(
                        protein_pos=batch_x,
                        protein_v=batch_h,
                        batch_protein=global_batch,
                        batch_ligand=batch_ligand,
                        sample_steps=model.cfg['evaluation']['sample_steps'],
                        n_nodes=batch_ligand.max().item() + 1,
                        desc=f'sample_batch_{i}'
                    )
        
        final = sample_chain[-1]  
        pred_pos, one_hot = final[0], final[1]

        pred_v = one_hot.argmax(dim=-1)  
        pred_atom_type = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in pred_v.tolist()]  
        
        num_molecules = batch_ligand.max().item() + 1
        for i in range(num_molecules):
            atom_index = (batch_ligand == i).cpu()
            atom_pos = (pred_pos).cpu()[atom_index]
            atom_type = (torch.tensor(pred_atom_type).cpu())[atom_index]
            output_mols.append({'x': atom_pos, 'h': atom_type})
        print(len(output_mols))
    builder = MoleculeBuilder()
    molecules = []
    sdf_path = os.path.join(save_dir, 'merged.sdf')
    with Chem.SDWriter(sdf_path) as writer:
        for mol in output_mols:
            rdkit_mol = builder.build_mol(mol['x'].numpy(), mol['h'].numpy())
            writer.write(rdkit_mol)
            molecules.append(rdkit_mol)

def main(args):
    set_random_seed(42)
    root_folder = args.root_folder  
    output_folder = args.pairs_folder      
    num_pairs = args.num_pairs   
    n_interpolations = args.n_interpolations  
    if root_folder != None:
        generate_and_save_molecule_pairs(root_folder, output_folder, num_pairs)
    pairs = load_molecule_pairs(output_folder)
    config = load_config(args.config)
    model = TrainLoop(config)
    model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
    model.to(args.device) 
    model.eval()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    device = next(model.parameters()).device
    all_interpolation_results = []
    for pair in tqdm(pairs, desc='Interpolating'):
        mol_start_data = pair['mol_start']
        mol_end_data = pair['mol_end']
        interpolation_results = interpolate_and_generate(model, mol_start_data, mol_end_data, n_interpolations)
        all_interpolation_results.append(interpolation_results)
    group_size = args.group_size  
    build_batch_and_sample(model, all_interpolation_results, group_size, save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process SDF files and perform molecule generation.')
    parser.add_argument('--root_folder', type=str, help='Root folder containing SDF files',default=None)
    parser.add_argument('--pairs_folder', type=str, help='Output folder to save processed molecules,NOTICE: this folder must be empty!',default='data/latent_experiment/ex2')
    parser.add_argument('--num_pairs', type=int, help='Number of molecule pairs to generate',default=1000)
    parser.add_argument('--n_interpolations', type=int, help='Number of interpolations per pair',default=12)
    parser.add_argument('--save_dir', type=str, help='Output folder to save generated molecules',default='./latent_experiment/ex2/generated')
    parser.add_argument('--ckpt_path', type=str,default='ckpt-zinc9M/model-epoch=24-val_loss=3.40.ckpt')
    parser.add_argument('--config', type=str,default='./config.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--group_size', type=int, default=50,help='NOTICE:group_size must be smaller than num_pairs!')
    args = parser.parse_args()
    if not os.path.exists(args.pairs_folder):
        os.makedirs(args.pairs_folder)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.group_size > args.num_pairs:
        raise ValueError('group_size must be smaller than num_pairs!')
    main(args)