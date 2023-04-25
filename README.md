# ThermoGNN

ThermoGNN is a computational biology tool to predict the changes in thermodynamic stability of protein structure
upon point mutations with **Siamese Graph Attention Network**. ThermoGNN constructs the residue interaction network
around the mutation site, and model on the impact of point mutation on the interactions among the neighborhood residues.
ThermoGNN integrates physicochemical properties of amino acids, multiple alignment profiles and energy scores.

## Installation

Several third-party software and python libraries are required to use ThermoGNN for thermodynamic stability predictions.
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

## Results

Performance comparison on Ssym dataset

| **Method**  | **$r_{dir}$** | **$r_{rev}$** | **$\sigma_{dir}$** | **$\sigma_{rev}$** | **$r_{dir-rev}$** |
| ----------- | --------------| --------------| -------------------| ------------------ | ------------------|
| ACDC-NN     | 0.57          | 0.57          | 1.45               | 1.45               | **-1.00**         |
| ThermoNet   | 0.47          | 0.47          | 1.56               | 1.55               | -0.96             |
| DDGun3D     | 0.56          | 0.53          | 1.42               | 1.46               | -0.99             |
| INPS        | 0.51          | 0.50          | 1.42               | 1.44               | -0.99             |
| PopMusicSym | 0.48          | 0.48          | 1.58               | 1.62               | -0.77             |
| SDM         | 0.51          | 0.32          | 1.74               | 2.28               | -0.75             |
| ThermoGNN   | **0.60**      | **0.60**      | **1.27**           | **1.27**           | **-1.00**         |

Performance comparison on p53 dataset

| **Method**  | **$r_{dir}$** | **$r_{rev}$** | **$\sigma_{dir}$** | **$\sigma_{rev}$** | **$r_{dir-rev}$** |
| ----------- | --------------| --------------| -------------------| ------------------ | ------------------|
| ACDC-NN    | **0.62**      | **0.61**         | **1.67**              | 1.72          | -0.99                        |
| ThermoNet  | 0.45          | 0.56             | 2.01                  | 1.92          | -0.93                        |
| ThermoGNN  | 0.53          | 0.53             | 1.72                  | **1.71**      | **-1.00**                    |

Performance comparison on myoglobin dataset

| **Method**  | **$r_{dir}$** | **$r_{rev}$** | **$\sigma_{dir}$** | **$\sigma_{rev}$** | **$r_{dir-rev}$** |
| ---------- | ------------------------ | ---------------- | --------------------- | ------------- | ---------------------------- |
| ACDC-NN    | 0.58                     | 0.57             | 0.89                  | 0.89          | -0.99                        |
| ThermoNet  | 0.38                     | 0.37             | 1.16                  | 1.18          | -0.97                        |
| ThermoGNN  | **0.61**                 | **0.62**         | **0.83**              | **0.82**      | **-1.00**                    |

