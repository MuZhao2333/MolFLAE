import copy
import numpy as np
import torch
from types import SimpleNamespace
from time import time
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_scatter import scatter_mean, scatter_sum
import os
from model.bfn4sbdd import BFN4SBDDScoreModel
from model.encoder import Encoder
import utils.atom_num as atom_num
import datetime
import json
from utils.train import get_optimizer, get_scheduler
import wandb
from utils.build_mol import MoleculeBuilder
from rdkit import Chem
import torch.nn as nn
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

#center_pos ligand
def center_pos(ligand_pos,batch_ligand, mode=True):
    if mode == False:
        offset = 0.0
        pass
    elif mode == True:
        offset = scatter_mean(ligand_pos, batch_ligand, dim=0)
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return ligand_pos, offset
def dict_to_namespace(d):
        if not isinstance(d, dict):
            return d
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})

class TrainLoop(pl.LightningModule):
    # def __init__(self, config: Config):
    #     super().__init__()
    #     self.cfg = config
    #     self.decoder = BFN4SBDDScoreModel(**self.cfg.decoder.todict())
    #     # [ time, h_t, pos_t, edge_index]
    #     self.train_losses = []
    #     self.save_hyperparameters(self.cfg.todict())
    #     self.time_records = np.zeros(6)
    #     self.log_time = False

    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        print(self.cfg)
        print("encoder")
        print(self.cfg['encoder_config'])
        self.encoder = Encoder(**self.cfg['encoder_config'])

        # For KL divergence
        self.Wh_mu = nn.Linear(self.cfg['encoder_config']['hidden_dim'], self.cfg['optimal_layer_config']['latent_dim'])
        self.Wh_log_var = nn.Linear(self.cfg['encoder_config']['hidden_dim'], self.cfg['optimal_layer_config']['latent_dim'])
        self.Wx_log_var = nn.Linear(self.cfg['encoder_config']['hidden_dim'], 1) # has to be isotropic gaussian to maintain equivariance

        self.decoder = BFN4SBDDScoreModel(**self.cfg['decoder_config'])
        self.time_records = np.zeros(6)
        self.encoder_config = self.cfg['encoder_config']
        self.decoder_config = self.cfg['decoder_config']
        
        self.train_losses = []

        self.log_time = False
        self.print_model_params()

        self.save_dir = cfg['evaluation']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        self.test_result=[]
        
    def print_model_params(self):
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        encoder_params = count_params(self.encoder)
        optimal_layer_params=count_params(self.Wh_log_var)+count_params(self.Wh_mu)+count_params(self.Wx_log_var)
        decoder_params = count_params(self.decoder)
        total_params = encoder_params + decoder_params+optimal_layer_params
        
        print("\n" + "="*50)
        print(f"Encoder params: {encoder_params/1e6:.2f}M")
        print(f"KL layer params: {optimal_layer_params/1e6:.2f}M")
        print(f"Decoder params: {decoder_params/1e6:.2f}M")
        print(f"total params: {total_params/1e6:.2f}M")
        print("="*50 + "\n")

    def forward(self, x):
        pass

    def encode(self,one_hot_h, x, batch_ligand,deterministic=False):
        #global_batch：[0,0,……(10),1,1,……(10),…………]
        #global_h：hidden_dim
        global_h, global_x, global_batch = self.encoder(one_hot_h, x, batch_ligand)
        
        #mu&var;global_h:hidden_dim→latent_dim
        Zh_mu = self.Wh_mu(global_h)
        Zh_log_var = -torch.abs(self.Wh_log_var(global_h))
        Zx_mu = global_x.clone()
        Zx_log_var = -torch.abs(self.Wx_log_var(global_h)).expand_as(Zx_mu)

        data_size = torch.unique(global_batch).size(0)

        Zh_kl_loss = -0.5 * torch.sum(1.0 + Zh_log_var - Zh_mu * Zh_mu - torch.exp(Zh_log_var)) / (data_size * Zh_mu.shape[-1])
        Zx_kl_loss = -0.5 * torch.sum(1.0 + Zx_log_var - 
                                      (Zx_mu * Zx_mu + torch.exp(Zx_log_var))/(self.cfg['train']['kl_loss']['sigma2'])**2) / (data_size * Zx_mu.shape[-1])
        
        #rsample
        Zh_sampled = Zh_mu if deterministic else Zh_mu + torch.exp(Zh_log_var / 2) * torch.randn_like(Zh_mu)
        Zx_sampled = Zx_mu if deterministic else Zx_mu + torch.exp(Zx_log_var / 2) * torch.randn_like(Zx_mu)
        
        return Zh_sampled, Zx_sampled,global_batch, Zh_kl_loss, Zx_kl_loss
    
    def training_step(self, batch, batch_idx):
        t1 = time()
        h = batch['h']
        x = batch['x']
        batch_ligand = batch['batch']
        # h: [N_node,1]
        # x: [N_node,3]
        # batch_ligand: [N_node]

        t2 = time()
        #no data augmentation
        # with torch.no_grad():
        #     # add noise to protein_pos
        #     protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std
        #     gt_protein_pos = batch.protein_pos + protein_noise
        #     # random rotation as data aug
        #     if self.cfg.train.random_rot:
        #         M = np.random.randn(3, 3)
        #         Q, __ = np.linalg.qr(M)
        #         Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
        #         gt_protein_pos = gt_protein_pos @ Q
        #         ligand_pos = ligand_pos @ Q

        num_graphs = batch_ligand.max().item() + 1

        x, _ = center_pos(
            ligand_pos=x,
            batch_ligand=batch_ligand,
            mode=self.cfg['decoder_config']['center_pos_mode']
        ) 

        t3 = time()
        # t：[num_graphs,1],[0,1)
        t = torch.rand(
            [num_graphs, 1], dtype=x.dtype, device=x.device
        ).index_select(
            0, batch_ligand
        )  # different t for different molecules.

        if not self.cfg['decoder_config']['use_discrete_t'] and not self.cfg['decoder_config']['destination_prediction']:
            t = torch.clamp(t, min=self.decoder.t_min)  # clamp t to [t_min,1]

        t4 = time()
        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K=self.cfg['encoder_config']['ligand_v_dim']

        one_hot_h=F.one_hot(h, K).float()  # [N, K]
        Zh_sampled, Zx_sampled, global_batch,Zh_kl_loss, Zx_kl_loss = self.encode(one_hot_h, x, batch_ligand,deterministic=False)
        # print(f"Zh_sampled_shape:{Zh_sampled.shape}")
        # print(f"Zx_sampled_shape:{Zx_sampled.shape}")
        c_loss, d_loss, discretised_loss = self.decoder.loss_one_step(
            t,
            protein_pos=Zx_sampled,
            protein_v=Zh_sampled,
            batch_protein=global_batch,
            ligand_pos=x,
            ligand_v=h,
            batch_ligand=batch_ligand,
        )
        
        # here the discretised_loss is close for current version.

        recon_loss = torch.mean(self.cfg['train']['recon_loss']['c_loss_weight'] *c_loss + self.cfg['train']['recon_loss']['d_loss_weight'] * d_loss + discretised_loss)
        kl_loss=torch.mean(self.cfg['train']['kl_loss']['Zh_kl_loss_weight'] *Zh_kl_loss + self.cfg['train']['kl_loss']['Zx_kl_loss_weight'] * Zx_kl_loss)
        loss=self.cfg['train']['recon_loss']['recon_loss_weight']*recon_loss+self.cfg['train']['kl_loss']['kl_loss_weight']*kl_loss

        wandb.log({
            'lr': self.get_last_lr(),
            'train_loss': loss.item(),
            'train_recon_loss': recon_loss.item(),
            'train_kl_loss':kl_loss.item()
        })
        t5 = time()
        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'train_loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss':kl_loss.item()
                # 'test':2
            },
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg['train']['batch_size'],
        )

        # check if loss is finite, skip update if not
        if not torch.isfinite(loss):
            return None
        self.train_losses.append(loss.clone().detach().cpu())

        t0 = time()

        if self.log_time:
            self.time_records = np.vstack((self.time_records, [t0, t1, t2, t3, t4, t5]))
            print(f'step total time: {self.time_records[-1, 0] - self.time_records[-1, 1]}, batch size: {num_graphs}')
            print(f'\tpl call & data access: {self.time_records[-1, 1] - self.time_records[-2, 0]}')
            print(f'\tunwrap data: {self.time_records[-1, 2] - self.time_records[-1, 1]}')
            print(f'\tadd noise & center pos: {self.time_records[-1, 3] - self.time_records[-1, 2]}')
            print(f'\tsample t: {self.time_records[-1, 4] - self.time_records[-1, 3]}')
            print(f'\tget loss: {self.time_records[-1, 5] - self.time_records[-1, 4]}')
            print(f'\tlogging: {self.time_records[-1, 0] - self.time_records[-1, 5]}')
        return loss

    def validation_step(self, batch, batch_idx):

        original_h = batch['h']
        original_x = batch['x']
        batch_ligand = batch['batch']

        x, _ = center_pos(
            ligand_pos=original_x,
            batch_ligand=batch_ligand,
            mode=self.cfg['decoder_config']['center_pos_mode']
        )
        
        num_graphs = batch_ligand.max().item() + 1
        t = torch.rand(
            [num_graphs, 1], dtype=x.dtype, device=x.device
        ).index_select(
            0, batch_ligand
        )
        
        if not self.cfg['decoder_config']['use_discrete_t'] and not self.cfg['decoder_config']['destination_prediction']:
            t = torch.clamp(t, min=self.decoder.t_min)

        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in original_h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K = self.cfg['encoder_config']['ligand_v_dim']
        one_hot_h = F.one_hot(h, K).float()  # [N, K]
        Zh_sampled, Zx_sampled, global_batch,Zh_kl_loss, Zx_kl_loss = self.encode(one_hot_h, x, batch_ligand,deterministic=True)

        # print(f"Zh_sampled_shape:{Zh_sampled.shape}")
        # print(f"Zx_sampled_shape:{Zx_sampled.shape}")
        c_loss, d_loss, discretised_loss = self.decoder.loss_one_step(
            t,
            protein_pos=Zx_sampled,
            protein_v=Zh_sampled,
            batch_protein=global_batch,
            ligand_pos=x,
            ligand_v=h,
            batch_ligand=batch_ligand,
        )
        
        # here the discretised_loss is close for current version.

        recon_loss = torch.mean(self.cfg['train']['recon_loss']['c_loss_weight'] *c_loss + self.cfg['train']['recon_loss']['d_loss_weight'] * d_loss + discretised_loss)
        kl_loss=torch.mean(self.cfg['train']['kl_loss']['Zh_kl_loss_weight'] *Zh_kl_loss + self.cfg['train']['kl_loss']['Zx_kl_loss_weight'] * Zx_kl_loss)
        loss=recon_loss+kl_loss

        generated_data = self.shared_sampling_step(batch, batch_idx, sample_num_atoms='ref', desc='Val',deterministic=True)

        molecule_builder = MoleculeBuilder()

        generated_h = torch.tensor(generated_data['h'], dtype=torch.long).to(x.device)
        generated_x = generated_data['x'].to(x.device)
        generated_batch = generated_data['batch'].to(x.device)

        unique_batches = torch.unique(batch_ligand)
        original_unique_list = unique_batches.tolist()

        similarities = []
        for batch_idx_val in unique_batches:
            original_mask = batch_ligand == batch_idx_val

            original_atoms = original_h[original_mask].cpu().numpy().tolist()
            original_coords = original_x[original_mask].cpu().numpy()

            generated_mask = generated_batch == batch_idx_val

            generated_atoms = generated_h[generated_mask].cpu().numpy().tolist()
            generated_coords = generated_x[generated_mask].cpu().numpy()

            original_mol = molecule_builder.build_mol(original_coords, original_atoms)

            generated_mol = molecule_builder.build_mol(generated_coords, generated_atoms)

            if original_mol is None or generated_mol is None:
                print(f"Warning: Invalid molecule for batch_idx_val {batch_idx_val.item()}")
                continue

            similarity = molecule_builder.compute_iou(original_mol, generated_mol)
            similarities.append(similarity)

            # self.generate_and_log_3d_structures(original_mol, generated_mol, caption_prefix=f"mol_{batch_idx_val.item()}_")

        if similarities:
            avg_similarity = torch.tensor(similarities).mean().item()
        else:
            avg_similarity = 0.0  

        wandb.log({
            'val_loss': loss.item(),
            'val_recon_loss': recon_loss.item(),
            'val_kl_loss':kl_loss.item(),
            'val_similarity': avg_similarity,
        })
        self.log_dict({
            'val_loss': loss.item(),
            'val_similarity': avg_similarity,
        },
            prog_bar=True, 
            logger=True,
            on_step=True,
            sync_dist=True,
            batch_size=self.cfg['evaluation']['batch_size'],
        )
        
        return loss

    def shared_sampling_step(self, batch, batch_idx, sample_num_atoms, desc='',deterministic=True):
        # here we need to sample the molecules in the validation step
        h = batch['h']
        x = batch['x']
        batch_ligand = batch['batch']

        num_graphs = batch_ligand.max().item() + 1  # B

        n_nodes = batch_ligand.size(0)  # N_lig

        x, offset = center_pos(
            ligand_pos=x,
            batch_ligand=batch_ligand,
            mode=True,
        ) 

        #ref
        if sample_num_atoms == 'prior':
            ligand_num_atoms = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg['data']['normalizer_dict']['pos'])
                ligand_num_atoms.append(atom_num.sample_atom_num(pocket_size).astype(int))
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(x.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=x.device)
        elif sample_num_atoms == 'ref':
            batch_ligand = batch_ligand
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(x.device)
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")

        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=x.device), 
            ligand_num_atoms.cumsum(dim=0)
        ])

        # TODO reuse for visualization and test

        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K=self.cfg['encoder_config']['ligand_v_dim']
        h=F.one_hot(h, K).float()  # [N, K]

        # Zh_sampled, Zx_sampled, global_batch,Zh_kl_loss, Zx_kl_loss = self.encode(one_hot_h, x, batch_ligand)
        global_nodes,global_position,global_batch,Zh_kl_loss, Zx_kl_loss= self.encode(h, x, batch_ligand,deterministic=deterministic)

        # print("global_nodes",global_nodes.shape)
        # print("global_position",global_position.shape)
        # print("global_batch",global_batch.shape)
        theta_chain, sample_chain, y_chain = self.decoder.sample(
            protein_pos=global_position,
            protein_v=global_nodes,
            batch_protein=global_batch,
            batch_ligand=batch_ligand,
            # n_nodes=n_nodes,
            sample_steps=self.cfg['evaluation']['sample_steps'],
            n_nodes=num_graphs,
            # ligand_pos=ligand_pos,  # for debug only
            desc=desc,
        )

        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot = final[0] + offset[batch_ligand], final[1]

        # pred_pos = pred_pos * torch.tensor(
        #     self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=x.device
        # )
        # out_batch = copy.deepcopy(batch)
        # out_batch.protein_pos = out_batch.protein_pos * torch.tensor(
        #     self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=x.device
        # )

        pred_v = one_hot.argmax(dim=-1) #pred_v=[0,1,2……]
        pred_atom_type = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in pred_v.tolist()] #pred_atom_type=[6,7,8……]

        # for visualization
        atom_type = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]  # List[int]
        atom_type = torch.tensor(atom_type, dtype=torch.long, device=x.device)  # [N_lig]  [0,1,2……]

        # pred_aromatic = trans.is_aromatic_from_index(
        #     pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        # ) # List[bool]

        # print('[DEBUG]', num_graphs, len(ligand_cum_atoms))

        out_data={'h':pred_atom_type,'x':pred_pos,'batch':batch_ligand}

        return out_data

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        self.log(
            "epoch_loss",
            epoch_loss,
            batch_size=self.cfg['train']['batch_size'],
        )
        self.train_losses = []
    
    
    def configure_optimizers(self):

        train_cfg = dict_to_namespace(self.cfg['train'])

        optimizer_cfg = dict_to_namespace(self.cfg['train']['optimizer'])
        self.optim = get_optimizer(optimizer_cfg, self)
        self.scheduler, self.get_last_lr = get_scheduler(train_cfg, self.optim)
    
        return {
            'optimizer': self.optim, 
            'lr_scheduler': self.scheduler,
            # 'monitor': 'val_loss',
        }

    def test_step(self, batch, batch_idx):
        original_x = batch['x']
        original_h = batch['h']
        original_batch = batch['batch']
        print(original_batch)

        x, _ = center_pos(
            ligand_pos=original_x,
            batch_ligand=original_batch,
            mode=self.cfg['decoder_config']['center_pos_mode']
        )

        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in original_h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K = self.cfg['encoder_config']['ligand_v_dim']
        one_hot_h = F.one_hot(h, K).float()  # [N, K]
        global_nodes, global_position, global_batch,_,_ = self.encode(one_hot_h, x, original_batch,deterministic=True)

        unique_batches = torch.unique(original_batch)
        original_unique_list = unique_batches.tolist()
        
        grouped_results = []
        for batch_idx_val in unique_batches:
            mask = original_batch == batch_idx_val
            grouped_results.append({
                'initial': {
                    'x': original_x[mask].cpu().numpy().tolist(),
                    'h': original_h[mask].cpu().numpy().tolist(),
                    'batch': original_batch[mask].cpu().numpy().tolist(),
                },
                'samples': []
            })

        self._save_global_position_as_xyz(global_position, original_batch, batch_idx)

        n_samples = self.cfg['evaluation']['num_samples']
        samples = [self.shared_sampling_step(batch, batch_idx,
                                            sample_num_atoms=self.cfg['evaluation']['sample_num_atoms'],
                                            desc=f'Test-{i}/{n_samples}',deterministic=True)
                    for i in range(n_samples)]

        for sample_data in samples:
            sample_batch = sample_data['batch']
            for batch_idx_val in torch.unique(sample_batch):
                mask = sample_batch == batch_idx_val
                idx = original_unique_list.index(batch_idx_val.item())
                
                h_data = sample_data['h']
                if isinstance(h_data, torch.Tensor):
                    h_selected = h_data[mask].cpu().numpy().tolist()
                else:
                    mask_indices = torch.where(mask)[0].tolist()
                    h_selected = [h_data[i] for i in mask_indices]
                
                grouped_results[idx]['samples'].append({
                    'x': sample_data['x'][mask].cpu().numpy().tolist(),
                    'h': h_selected,
                    'batch': sample_batch[mask].cpu().numpy().tolist(),
                })
        
        if not hasattr(self, 'test_results') or self.test_results is None:
            self.test_results = []
        self.test_results.extend(grouped_results)
        return grouped_results
    
    def _save_global_position_as_xyz(self, global_position, batch, batch_idx):
        print(f"global_position shape: {global_position.shape}")
        print(f"global_position: {global_position}")
        
        all_pos = []

        if len(global_position.shape) == 1:
            global_position = global_position.reshape(-1, 3)
        
        all_pos = global_position.cpu().numpy()

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.save_dir, f'global_position_all_{timestamp}.xyz')

        with open(filename, 'w') as f:
            f.write(f"{len(all_pos)}\n")
            f.write(f"Global Position for All Batches\n")
            for (x, y, z) in all_pos:
                f.write(f"C {x} {y} {z}\n")
        
        print(f"Global position saved to: {filename}")

    def on_test_epoch_end(self):
        all_results = self.test_results
        
        processed_results = []
        for mol_group in all_results:
            processed_mol = {
                'initial': mol_group['initial'],
                'samples': mol_group['samples']
            }
            processed_results.append(processed_mol)
        
        self._save_results(processed_results)
        wandb.save(os.path.join(self.save_dir, f'test_results_*.json'))
    
    def _convert_tensor_to_dict(self, data_dict):
        return {
            'x': data_dict['x'].cpu().numpy().tolist(),
            'h': data_dict['h'],  
            'batch': data_dict['batch'].cpu().numpy().tolist()
        }
    
    def _save_results(self, results):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(self.save_dir, f'test_results_{timestamp}.json')
        
        output = {
            'config': self.cfg,
            'results': results,
            'save_time': timestamp,
            'num_molecules': len(results),
            'num_samples_per_mol': len(results[0]['samples']) if results else 0
        }
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\save to: {save_path}")