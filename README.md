# UCL-GLGNN
# Important: This code is for review only.
UCL-GLGNN is a computational biology tool to predict the changes in thermodynamic stability of protein structure
upon point mutations with **Siamese Graph Attention Network**. ThermoGNN constructs the residue interaction network
around the mutation site, and model on the impact of point mutation on the interactions among the neighborhood residues.
ThermoGNN integrates physicochemical properties of amino acids, multiple alignment profiles and energy scores.

## Installation

Several third-party software and python libraries are required to use UCL-GLGNN for thermodynamic stability predictions.
We outline the steps to install them in this section.

### Install Rosetta

Apply for a License and download Rosetta 3.12 from [https://www.rosettacommons.org/software/license-and-download](https://www.rosettacommons.org/software/license-and-download).

### Install HH-suite3 & Database

It is recommended to download HH-suite3 via conda, and the Uniclust30 database can be downloaded at
[http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/](http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/).

```shell
conda install -c conda-forge -c bioconda hhsuite
```

### Install PyTorch & PyTorch Geometrics

It is recommended to use [Anaconda](https://www.anaconda.com/products/individual)
to install PyTorch, PyTorch Geometrics and other required Python libraries.
If you want to train the model on GPU, the version of CUDA mut

```shell
conda create -n ThermoGNN python=3.9
source activate ThermoGNN
conda install pytorch torchvision torchaudio cudatoolkit=$YOUR_CUDA_VERSION -c pytorch
conda install pytorch-geometric -c rusty1s -c conda-forge
conda install -c bioconda biopython
pip install -r requirements.txt
pip install wandb # for visualization
```

## Usage
1. Use Rosetta to refine the structures.
   ```shell
   # relax.sh
   python ThermoGNN/tools/relax.py -i input-pdb \
                                   -l mutant_list.txt \
                                   --rosetta-bin relax.static.linuxgccrelease \
                                   -o data/pdbs/demo/
   ```
   `input-pdb` denotes the directory storing your prepared single-chain pdb structures.
   
    `mutation_list.txt` records the mutations in which each line is in the format of `1a23A 51 H L`.

2. Generate MSA profiles by hhblits.
    ```shell
    # hhblits.sh
    for pdb_dir in data/pdbs/demo/*
    do
      [[ -e $pdb_dir ]]
      python ThermoGNN/tools/hhblits.py -i $pdb_dir \
                                        -db hhsuite_db/UniRef30_2020_06 \
                                        -o data/hhm/demo/ \
                                        --cpu 40
    done
    ```
    `hhsuite_db/UniRef30_2020_06` is the path to the hhsuite database downloaded before.

3. Generate residue interaction networks, and predict the ddG of candidate proteins and mutations.
    ```shell
    python predict.py -l mutant_list.txt \
                      --model GAT \
                      --split demo # .pdb in data/pdbs/demo/, .hhm in data/hhm/demo/
    ```
   You can view the predicted ddGs in `prediction.csv` in default.

4. Train your own ThermoGNN model.

    You can modify the hyperparameters in `run.sh`, and train the new model. 

