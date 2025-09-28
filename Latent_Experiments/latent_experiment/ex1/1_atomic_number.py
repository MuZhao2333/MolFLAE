import os
import sys
import argparse
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
import torch
import numpy as np
from rdkit import Chem
from model.train_loop import TrainLoop
from config.config import load_config
from model.train_loop import MAP_ATOM_TYPE_ONLY_TO_INDEX, MAP_INDEX_TO_ATOM_TYPE_ONLY, center_pos
import torch.nn.functional as F
from utils.build_mol import MoleculeBuilder
from rdkit import Chem
import torch
from tqdm import tqdm
import random
import time
import pytorch_lightning as pl
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
MAP_ATOM_TYPE_TO_ATOM_SIGN = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    15: 'P',
    16: 'S',
    17: 'Cl',
    35: 'S',
    53: 'I',
}
def process_sdf_files_to_list(root_folder):
    """
    Recursively find all sdf files in the specified folder, process each sdf file,
    and return a list containing processing results of all molecules.
    If a molecule contains atoms not in the MAP_ATOM_TYPE_ONLY_TO_INDEX dictionary, it will be discarded.

    Args:
        root_folder (str): The path of the root folder to search.

    Returns:
        list: A list containing processing results of each molecule, where each element is a dictionary
              with keys like 'h', 'x', 'atom_num', etc.
    """
    result_list = []
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

                        # Check if atom types in the molecule are all in MAP_ATOM_TYPE_ONLY_TO_INDEX
                        valid_molecule = all(atomic_num in MAP_ATOM_TYPE_ONLY_TO_INDEX for atomic_num in atom_types)

                        if valid_molecule:
                            conformer = mol.GetConformer()
                            positions = conformer.GetPositions()
                            atom_indices = [MAP_ATOM_TYPE_ONLY_TO_INDEX[atomic_num] for atomic_num in atom_types]

                            result_list.append({
                                'h': torch.tensor(atom_indices, dtype=torch.long),
                                'x': torch.tensor(positions, dtype=torch.float32),
                                'atom_num': len(atom_indices)
                            })
                        else:
                            print(f"Discarded molecule with invalid atom types: {mol.GetProp('_Name')}")
                        progress_bar.update(1)

    progress_bar.close()
    return result_list
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
        torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # <url id="d0n990vo7p5ti8ov5rv0" type="url" status="parsed" title="torch.use_deterministic_algorithms — PyTorch 2.7 documentation" wc="6723">https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html</url>
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # avoiding nondeterministic algorithms (see <url id="d0n990vo7p5ti8ov5rvg" type="url" status="parsed" title="Reproducibility — PyTorch 2.7 documentation" wc="7973">https://pytorch.org/docs/stable/notes/randomness.html</url>)
    torch.use_deterministic_algorithms(True)

def main():
    set_random_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf_folder', type=str,default='./data/latent_experiment/val')
    parser.add_argument('--output_folder', type=str,default='./latent_experiment/ex1/output')
    parser.add_argument('--ckpt_path', type=str,default='ckpt-zinc9M/model-epoch=24-val_loss=3.40.ckpt')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    batch_size = args.batch_size
    mol_1000=process_sdf_files_to_list(args.sdf_folder)
    all_h = torch.cat([entry['h'] for entry in mol_1000], dim=0)
    all_x = torch.cat([entry['x'] for entry in mol_1000], dim=0)
    all_batch = []
    current_index = 0
    for entry in mol_1000:
        atom_num = entry['atom_num']
        all_batch += [current_index] * atom_num
        current_index += 1
    all_batch = torch.tensor(all_batch, dtype=torch.long)
    # print(all_h.shape,all_x.shape,all_batch.shape)
    # print(all_h)
    save_dir=args.output_folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config = load_config(args.config)
    model = TrainLoop(config)
    model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
    model.to(args.device)
    model.eval()
    device = next(model.parameters()).device

    h, x, batch_ligan_orig = all_h.to(device), all_x.to(device), all_batch.to(device)

    x, _ = center_pos(x, batch_ligan_orig, mode=True)

    K = model.cfg['encoder_config']['ligand_v_dim']
    one_hot_h1 = F.one_hot(h, K).float()

    with torch.no_grad():
        Zh, Zx, global_batch, _, _ = model.encode(one_hot_h1, x, batch_ligan_orig, deterministic=True)
        # print("encode result")
        # print(Zh.shape,Zx.shape,global_batch.shape)
        # decode with atom number variation
        mol_atom_nums = [entry['atom_num'] for entry in mol_1000]
        edit_batches = {}
        start_indices = [0]
        for atom_num in mol_atom_nums[:-1]:
            start_indices.append(start_indices[-1] + atom_num)
        # number of atoms to add or delete
        n_max=2
        for n in range(-n_max, n_max + 1):
            edit_batch = []
            current_index = 0

            for i, atom_num in enumerate(mol_atom_nums):
                new_atom_num = atom_num + n

                new_atom_num = max(new_atom_num, 1)

                edit_batch += [current_index] * new_atom_num
                current_index += 1

            edit_batches[n] = torch.tensor(edit_batch, dtype=torch.long).to(device)

        # print(edit_batches[0].shape,edit_batches[-1].shape,edit_batches[1].shape,edit_batches[-2].shape,edit_batches[2].shape)
        similarity_dict = {}
        
        for n in range(-n_max, n_max + 1):
            current_edit_batch = edit_batches[n]
            unique_molecules = torch.unique(global_batch)
            num_batches = (len(unique_molecules) + batch_size - 1) // batch_size

            all_pred_pos = []
            all_pred_atom_type = []
            all_batch_indices = []

            for batch_idx in range(num_batches):

                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(unique_molecules))

                current_molecule_ids = unique_molecules[start_idx:end_idx]

                global_mask = torch.isin(global_batch, current_molecule_ids)
                ligand_mask = torch.isin(current_edit_batch, current_molecule_ids)
                current_Zx = Zx[global_mask]
                current_Zh = Zh[global_mask]
                current_global_batch = global_batch[global_mask]
                current_edit_batch_filtered = current_edit_batch[ligand_mask]

                _, current_global_batch_remapped = torch.unique(current_global_batch, return_inverse=True)
                current_edit_batch_remapped = current_edit_batch_filtered.clone()
                for mol_id in range(len(current_molecule_ids)):
                    current_edit_batch_remapped[current_edit_batch_filtered == current_molecule_ids[mol_id]] = mol_id
                print(current_Zx.shape,current_Zh.shape,current_global_batch_remapped.shape,current_edit_batch_remapped.shape)
                print(current_edit_batch_remapped.max().item() + 1)

                theta_chain, sample_chain, y_chain = model.decoder.sample(
                    protein_pos=current_Zx,
                    protein_v=current_Zh,
                    batch_protein=current_global_batch_remapped,
                    batch_ligand=current_edit_batch_remapped,
                    sample_steps=model.cfg['evaluation']['sample_steps'],
                    n_nodes=current_edit_batch_remapped.max().item() + 1,
                    desc=f'sample_{n}_batch_{batch_idx}'
                )

                final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
                pred_pos, one_hot = final[0], final[1]

                pred_v = one_hot.argmax(dim=-1)  # pred_v=[0,1,2……]
                pred_atom_type = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in pred_v.tolist()]  # pred_atom_type=[6,7,8……

                all_pred_pos.extend(pred_pos.cpu().numpy())
                all_pred_atom_type.extend(pred_atom_type)

                batch_indices = torch.repeat_interleave(
                    torch.arange(len(current_molecule_ids), device=device),
                    torch.bincount(current_edit_batch_remapped)
                )
                all_batch_indices.extend(batch_indices.cpu().numpy())

            pred_pos = np.array(all_pred_pos)
            pred_atom_type = np.array(all_pred_atom_type)
            batch_indices = np.array(all_batch_indices)
            molecule_builder = MoleculeBuilder()

            # generated_h = torch.tensor(generated_data['h'], dtype=torch.long).to(x.device)
            # generated_x = generated_data['x'].to(x.device)
            # generated_batch = generated_data['batch'].to(x.device)

            unique_batches = torch.unique(edit_batches[n])
            original_unique_list = unique_batches.tolist()

            similarities = []
            for batch_idx_val in unique_batches:
                time0=time.time()

                original_mask = (edit_batches[0] == batch_idx_val)
                original_atoms = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in h[original_mask].cpu().numpy().tolist()]
                original_coords = x[original_mask].cpu().numpy()

                # original_atoms = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in mol_1000[batch_idx_val]['h'].cpu().numpy().tolist()]
                # original_coords = mol_1000[batch_idx_val]['x'].cpu().numpy()

                generated_mols_mask = (edit_batches[n] == batch_idx_val).cpu()

                # print(generated_mols_mask.device)
                # print(pred_atom_type.device)
                # print(pred_pos.device)
                generated_atoms = pred_atom_type[generated_mols_mask].tolist()

                generated_coords = pred_pos[generated_mols_mask]

                time1=time.time()
                original_mol = molecule_builder.build_mol(original_coords, original_atoms)
                time2=time.time()
                generated_mol = molecule_builder.build_mol(generated_coords, generated_atoms)
                time3=time.time()

                orig_filename = os.path.join(save_dir, f'{batch_idx_val}_atm{n}_ori.sdf')
                with Chem.SDWriter(orig_filename) as writer:
                    writer.write(original_mol)
                generate_filename = os.path.join(save_dir, f'{batch_idx_val}_atm{n}_gen.sdf')
                with Chem.SDWriter(generate_filename) as writer:
                    writer.write(generated_mol)
                time4=time.time()
                if original_mol is None or generated_mol is None:
                    print(f"Warning: Invalid molecule for atm_{n}_{batch_idx}_{batch_idx_val}")
                    continue

                similarity = molecule_builder.compute_iou(original_mol, generated_mol)
                print(f'Similarity for batch atm_{n}_{batch_idx}_{batch_idx_val}: {similarity} \nbuil1:{time2-time1} \nbuil2:{time3-time2} \nsave:{time4-time3}',flush=True)
                similarities.append(similarity)

            if similarities:
                avg_similarity = torch.tensor(similarities).mean().item()
            else:
                avg_similarity = 0.0
            similarity_dict[n]= avg_similarity

        for n in range(-n_max, n_max + 1):
            print(f"Similarity for n={n}: {similarity_dict[n]}")


if __name__ == '__main__':
    main()