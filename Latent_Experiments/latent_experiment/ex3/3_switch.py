import sys
import os
import argparse
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
import numpy as np
from rdkit import Chem

from model.train_loop import TrainLoop,center_pos
from config.config import load_config
import torch
import pytorch_lightning as pl
import random
import torch.nn.functional as F
from tqdm import tqdm
from utils.build_mol import MoleculeBuilder
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

    if num_valid_molecules < 2:
        return

    molecule_indices = list(range(num_valid_molecules))
    generated_pairs = set()  

    for i in range(num_pairs):
        while True:
            index1 = random.randint(0, num_valid_molecules - 1)
            index2 = random.randint(0, num_valid_molecules - 1)
            if index1 != index2:
                min_index, max_index = sorted([index1, index2])
                pair_key = (min_index, max_index)
                if pair_key not in generated_pairs:
                    generated_pairs.add(pair_key)
                    save_molecule_pairs(output_folder, valid_molecules[index1], valid_molecules[index2], i + 1)
                    break


def load_molecule_pairs(folder_path):

    molecule_pairs = []
    start_files = sorted([f for f in os.listdir(folder_path) if f.endswith('_start.sdf')])
    
    for start_file in start_files:
        index = start_file.split('_')[0]
        end_file = f"{index}_end.sdf"
        if not os.path.exists(os.path.join(folder_path, end_file)):
            continue
        start_mol = Chem.SDMolSupplier(os.path.join(folder_path, start_file))[0]
        if start_mol is None:
            continue
        end_mol = Chem.SDMolSupplier(os.path.join(folder_path, end_file))[0]
        if end_mol is None:
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
    device=model.device
    x_start=x_start.to(device)
    x_end=x_end.to(device)
    h_start=torch.tensor([MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h_start]).to(device)
    h_end=torch.tensor([MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h_end]).to(device)

    one_hot_h_start = F.one_hot(h_start, K).float().to(device)
    one_hot_h_end = F.one_hot(h_end, K).float().to(device)

    batch_ligan_orig_start = torch.zeros_like(h_start).to(device)
    batch_ligan_orig_end = torch.zeros_like(h_end).to(device)
    x_start, _ = center_pos(x_start, batch_ligan_orig_start, mode=True)
    x_end, _ = center_pos(x_end, batch_ligan_orig_end, mode=True)
    # print(h_start.device)
    # print(one_hot_h_start.device)
    # print(batch_ligan_orig_start.device)
    with torch.no_grad():
        Zh_start, Zx_start, _, _, _ = model.encode(one_hot_h_start, x_start, batch_ligan_orig_start, deterministic=True)
        Zh_end, Zx_end, _, _, _ = model.encode(one_hot_h_end, x_end, batch_ligan_orig_end, deterministic=True)

    alpha_values = torch.linspace(0, 1, n_interpolations)
    Zh_interp = torch.stack([Zh_start + alpha * (Zh_end - Zh_start) for alpha in alpha_values])
    Zx_interp = torch.stack([Zx_start + alpha * (Zx_end - Zx_start) for alpha in alpha_values])

    atom_num_start = mol_start_data['atom_num']
    atom_num_end = mol_end_data['atom_num']
    # print(type(atom_num_start), atom_num_start)
    # print(type(atom_num_end), atom_num_end)
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
        # torch seed init.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.
    os.environ['PYTHONHASHSEED'] = str(seed)
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' 
    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)

def main(args):
    set_random_seed(42)
    torch.set_printoptions(threshold=2, precision=2)
    root_folder = args.root_folder
    output_folder = args.pairs_folder
    num_pairs = args.num_pairs
    n_interpolations= 2 # reuse ex2
    if root_folder !=None:
        generate_and_save_molecule_pairs(root_folder, output_folder, num_pairs)
    pairs=load_molecule_pairs(output_folder)
    print(pairs[0])
    config = load_config(args.config)
    model = TrainLoop(config)
    global_nodes=10
    model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
    model.to(args.device) 
    model.eval()
    device = next(model.parameters()).device
    all_interpolation_results = []
    save_dir=args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for pair in tqdm(pairs,desc='Interpolating'):
        mol_start_data = pair['mol_start']
        mol_end_data = pair['mol_end']
        interpolation_results = interpolate_and_generate(model, mol_start_data, mol_end_data, n_interpolations)
        all_interpolation_results.append(interpolation_results)
    # print(len(all_interpolation_results))
    # print(all_interpolation_results[1])
    batch_ligand=[]
    global_batch=[]
    Zh0_all=[]
    Zx0_all=[]
    Zh1_all=[]
    Zx1_all=[]
    print(len(all_interpolation_results))
    for i in range(len(all_interpolation_results)):
        Zh0=all_interpolation_results[i][0]['Zh']
        Zx0=all_interpolation_results[i][0]['Zx']

        Zh1=all_interpolation_results[i][1]['Zh']
        Zx1=all_interpolation_results[i][1]['Zx']

        atom_num=all_interpolation_results[i][0]['atom_num']
        Zh0_all.append(Zh0)
        Zx0_all.append(Zx0)
        Zh1_all.append(Zh1)
        Zx1_all.append(Zx1)
        batch_ligand.append((torch.ones(atom_num)*i).int())
        global_batch.append((torch.ones(global_nodes)*i).int())
    Zh0_all=torch.cat(Zh0_all, dim=0).to(device)
    Zx0_all=torch.cat(Zx0_all, dim=0).to(device)
    Zh1_all=torch.cat(Zh1_all, dim=0).to(device)
    Zx1_all=torch.cat(Zx1_all, dim=0).to(device)
    batch_ligand=torch.cat(batch_ligand, dim=0).to(device)
    global_batch=torch.cat(global_batch, dim=0).to(device)
    print(batch_ligand)
    original_swZh=[]
    switchZh_output=[]
    original_swZx=[]  
    switchZx_output=[]
    donor_swZh=[]
    donor_swZx=[]
    with torch.no_grad():
        theta_chain, sample_chain, y_chain = model.decoder.sample(
            protein_pos=Zx0_all,
            protein_v=Zh1_all,
            batch_protein=global_batch,
            batch_ligand=batch_ligand,
            sample_steps=model.cfg['evaluation']['sample_steps'],
            n_nodes=batch_ligand.max().item() + 1,
            desc=f'sample_swZh'
        )
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot = final[0], final[1]

        pred_v = one_hot.argmax(dim=-1)  # pred_v=[0,1,2……]
        pred_atom_type = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in pred_v.tolist()]  # pred_atom_type=[6,7,8……
        
        num_molecules=batch_ligand.max().item()+1
        for i in range(num_molecules):
            atom_index=(batch_ligand==i).cpu()
            # print(atom_index.device)
            # print(pred_pos.device)
            # print(pred_atom_type.device)
            atom_pos=(pred_pos).cpu()[atom_index]
            atom_type=(torch.tensor(pred_atom_type).cpu())[atom_index]
            switchZh_output.append({'x':atom_pos, 'h':atom_type})
            original_x=pairs[i]['mol_start']['x']
            original_h=pairs[i]['mol_start']['h']
            donor_x=pairs[i]['mol_end']['x']
            donor_h=pairs[i]['mol_end']['h']
            original_x,_=center_pos(original_x,(torch.zeros_like(original_h)).long())
            donor_x,_=center_pos(donor_x,(torch.zeros_like(donor_h)).long())
            original_swZh.append({'x':original_x, 'h':original_h})
            donor_swZh.append({'x':donor_x, 'h':donor_h})
            # print("one_inter",one_interpolation)
        print(len(switchZh_output))
        builder=MoleculeBuilder()
        molecules = []
        sdf_path=save_dir+'/switchZh.sdf'
        with Chem.SDWriter(sdf_path) as writer:
            for i in range(num_molecules):
                original_mol=builder.build_mol(original_swZh[i]['x'].numpy(), original_swZh[i]['h'].numpy())
                # Chem.SanitizeMol(rdkit_mol)
                writer.write(original_mol)
                donor_mol=builder.build_mol(donor_swZh[i]['x'].numpy(), donor_swZh[i]['h'].numpy())
                # Chem.SanitizeMol(rdkit_mol)
                writer.write(donor_mol)
                switchZh_mol=builder.build_mol(switchZh_output[i]['x'].numpy(), switchZh_output[i]['h'].numpy())
                # Chem.SanitizeMol(rdkit_mol)
                writer.write(switchZh_mol)
    with torch.no_grad():
        theta_chain, sample_chain, y_chain = model.decoder.sample(
            protein_pos=Zx1_all,
            protein_v=Zh0_all,
            batch_protein=global_batch,
            batch_ligand=batch_ligand,
            sample_steps=model.cfg['evaluation']['sample_steps'],
            n_nodes=batch_ligand.max().item() + 1,
            desc=f'sample_swZx'
        )
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot = final[0], final[1]

        pred_v = one_hot.argmax(dim=-1)  # pred_v=[0,1,2……]
        pred_atom_type = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in pred_v.tolist()]  # pred_atom_type=[6,7,8……
        
        num_molecules=batch_ligand.max().item()+1
        for i in range(num_molecules):
            atom_index=(batch_ligand==i).cpu()
            # print(atom_index.device)
            # print(pred_pos.device)
            # print(pred_atom_type.device)
            atom_pos=(pred_pos).cpu()[atom_index]
            atom_type=(torch.tensor(pred_atom_type).cpu())[atom_index]
            switchZx_output.append({'x':atom_pos, 'h':atom_type})

            original_x=pairs[i]['mol_start']['x']
            original_h=pairs[i]['mol_start']['h']
            donor_x=pairs[i]['mol_end']['x']
            donor_h=pairs[i]['mol_end']['h']
            original_x,_=center_pos(original_x,(torch.zeros_like(original_h)).long())
            donor_x,_=center_pos(donor_x,(torch.zeros_like(donor_h)).long())
            original_swZx.append({'x':original_x, 'h':original_h})
            donor_swZx.append({'x':donor_x, 'h':donor_h})
            # print("one_inter",one_interpolation)
        print(len(switchZx_output))
        builder=MoleculeBuilder()
        sdf_path= save_dir+'/switchZx.sdf'
        with Chem.SDWriter(sdf_path) as writer:
            for i in range(num_molecules):
                original_mol=builder.build_mol(original_swZx[i]['x'].numpy(), original_swZx[i]['h'].numpy())
                # Chem.SanitizeMol(rdkit_mol)
                writer.write(original_mol)
                donor_mol=builder.build_mol(donor_swZx[i]['x'].numpy(), donor_swZx[i]['h'].numpy())
                # Chem.SanitizeMol(rdkit_mol)
                writer.write(donor_mol)
                switchZx_mol=builder.build_mol(switchZx_output[i]['x'].numpy(), switchZx_output[i]['h'].numpy())
                # Chem.SanitizeMol(rdkit_mol)
                writer.write(switchZx_mol)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process SDF files and perform molecule generation.')
    parser.add_argument('--root_folder', type=str, help='Root folder containing SDF files', default=None)
    parser.add_argument('--pairs_folder', type=str, help='Output folder to save processed molecules, NOTICE: this folder must be empty!', default='data/latent_experiment/ex3')
    parser.add_argument('--num_pairs', type=int, help='Number of molecule pairs to generate', default=1000)
    parser.add_argument('--save_dir', type=str, help='Output folder to save generated molecules', default='./latent_experiment/ex3/output')
    parser.add_argument('--ckpt_path', type=str, default='ckpt-zinc9M/model-epoch=24-val_loss=3.40.ckpt')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.pairs_folder):
        os.makedirs(args.pairs_folder)

    main(args)