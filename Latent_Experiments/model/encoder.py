import torch
import torch.nn as nn
from utils.common import MLP
from model.uni_transformer import UniTransformerO2TwoUpdateGeneral


class Encoder(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, knn=32,
                 num_r_gaussian=20, edge_feat_dim=4, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='global', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False,
                 global_node_num=10,ligand_v_dim=9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.global_node_num = global_node_num
        self.ligand_v_dim=ligand_v_dim
        self.input_adaptor=MLP(in_dim=ligand_v_dim, out_dim=hidden_dim, hidden_dim=hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False)
        self.unet = UniTransformerO2TwoUpdateGeneral(
            num_blocks=num_blocks,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            knn=knn,
            num_r_gaussian=num_r_gaussian,
            edge_feat_dim=edge_feat_dim,
            num_node_types=num_node_types,
            act_fn=act_fn,
            norm=norm,
            cutoff_mode=cutoff_mode,
            ew_net_type=ew_net_type,
            num_init_x2h=num_init_x2h,
            num_init_h2x=num_init_h2x,
            num_x2h=num_x2h,
            num_h2x=num_h2x,
            r_max=r_max,
            x2h_out_fc=x2h_out_fc,
            sync_twoup=sync_twoup
        )
        #learnable global_h layer
        self.global_h = nn.Parameter(torch.randn(self.global_node_num, self.hidden_dim))
        nn.init.normal_(self.global_h, mean=0, std=0.1) 
    def __repr__(self):
        return (
            f"Encoder:\n{self.unet}\n"
        )
    def forward(self, h, x, batch):
        # h：[num_nodes, ligand_v_dim]
        # x：[num_nodes, 3]
        # mask_ligand：[num_nodes]
        # batch：[num_nodes]
        # Add global nodes
        batch_size = batch.max().item() + 1  # number of molecules
        # nn.par → fc → cat
        # global_h = torch.zeros(
        #     (batch_size * self.global_node_num, self.ligand_v_dim), 
        #     device=h.device
        # )
        
        # 0 init
        global_x = torch.zeros(
            (batch_size * self.global_node_num, 3), 
            device=x.device
        )
        
        global_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=batch.device), 
            repeats=self.global_node_num
        )
        # h_new = torch.cat([h, global_h], dim=0)
        x_new = torch.cat([x, global_x], dim=0)
        batch_new = torch.cat([batch, global_batch], dim=0)

        # generate mask: normal_nodes=0,global_nodes=1
        mask_ligand_new = torch.cat([
            torch.zeros(len(h), dtype=torch.bool, device=h.device), 
            torch.ones(batch_size * self.global_node_num, dtype=torch.bool, device=h.device) 
        ], dim=0)

        # Update h and x using the network
        # ligand_v_dim → hidden_dim
        h=self.input_adaptor(h)
        global_h=self.global_h.repeat(batch_size, 1)
        h_new = torch.cat([h, global_h], dim=0)
        # print("#####################GNN#################")
        # print(f'h_new:{h_new}')
        # print(f'x_new:{x_new}')
        # print(f'mask_ligand_new:{mask_ligand_new}')
        # print(f'batch_new:{batch_new}')
        # print("#####################GNN#################")
        outputs = self.unet(h_new, x_new, mask_ligand_new, batch_new,return_edge=True)
        # print(f'output:{outputs}')
        h_updated = outputs['h']
        x_updated = outputs['x']

        # Extract global nodes
        global_nodes_updated = h_updated[-batch_size * self.global_node_num:]
        global_positions_updated = x_updated[-batch_size * self.global_node_num:]

        return global_nodes_updated,global_positions_updated,global_batch

        