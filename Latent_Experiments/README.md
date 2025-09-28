# Latent Space Experiments
Before running the experiments, please download the pre-trained model from [Google Drive](abaaba) and place it in the `Latent_Experiments/ckpt-zinc9M` directory, as well as the sdf files from [Google Drive](abaaba) and place it in the `Latent_Experiments/data` directory.

As we observed when running the experiments on a single NVIDIA A100, the maximum memory usage was approximately 22GB. Therefore, we recommend running the experiments on a machine with at least 24GB of GPU memory.

MolFLAE only output atom point clouds, and the bond information is inferred using OpenBabel. Due to some bugs in OpenBabel, we need to repair the bond orders of the generated molecules using Schrodinger in some experiments.

## 1. Generating Analogs with Different Atom Numbers
Reproduce atom number edit experiment with the same 932 sdf files we used in the paper:
```bash
cd Latent_Experiments
python latent_experiment/ex1/1_atomic_number.py
```
The output sdf files will be saved in `Latent_Experiments/latent_experiment/ex1/output`.

## 2. Exploring the disentanglement of the latent space via molecule reconstruction
Use the 1000 pairs we selected:
```bash
python latent_experiment/ex2/2_inter.py
```
The generated sdf files will be saved in `Latent_Experiments/latent_experiment/ex2/generated`.

Then, repaire the molecules using Schrodinger, for example:
```bash
/schrodinger2024-1/run fix_bond_orders.py -process_all_residues input.sdf output.sdf
```
Then compute the metrics:
```bash
python latent_experiment/ex2/2_compute.py
```

## 3. Latent Interpolation
Use the 1000 pairs we selected:
```bash
cd Latent_Experiments
python latent_experiment/ex3/3_switch.py
```
The generated sdf files will be saved in `Latent_Experiments/latent_experiment/ex3/output`.
Then, repaire the molecules using Schrodinger, for example:
```bash
/schrodinger2024-1/run fix_bond_orders.py -process_all_residues input.sdf output.sdf
```
Then compute the metrics:
```bash
python latent_experiment/ex3/3_compute.py --switch zh  # compute metrics of switchZh molecules  
python latent_experiment/ex3/3_compute.py --switch zx  # compute metrics of switchZx molecules  
``` 

其他选项，如用自己的sdf做实验1，或是选其他的pairs复现实验2和3，根据代码从命令行输入即可（
If you want to use your own sdf files for experiment 1, or select other pairs to reproduce experiments 2 and 3, you can input them from the command line according to the code.