# Code for Benchmarking Computational Methods for Emerging Drug-Drug Interaction Prediction


## Installation
### 1. Set Up Environment for `DDI_ben`, `TextDDI` and `DDI-GPT`
The models MLP, MSTE, Decagon, SSI-DDI, MRCGNN, SAGAN and TIGER in `DDI_ben` and `TextDDI`, `DDI-GPT` share the same environment. Our running environment is a Linux server with Ubuntu. You can set up the environment as follows:

```bash
# Create and activate Conda environment
conda create -n ddibench python=3.8.0
conda activate ddibench

# Install dependencies
# We provide the exact versions of packages we use
pip install -r DDI_ben/requirements.txt
```

### 2. Set Up Environment for `EmerGNN`
`EmerGNN` require different environments. It should be set up separately according to their respective official repositories. You can find the official repositories here:  
- `EmerGNN`: [EmerGNN Repository](https://github.com/LARS-research/EmerGNN)  

## Running the Code
First, `cd` into the corresponding directory, i.e., DDI_ben, TextDDI, EmerGNN/Drugbank, EmerGNN/TWOSIDES or SumGNN. After that,

- For `DDI_ben`, you can run the code as follows:
```bash
python main.py --model MSTE  --dataset drugbank --dataset_type random  --lr 3e-3 --gpu 0 
```

- For `TextDDI`, 
```bash
python drugbank/main_drugbank.py --dataset_type finger --gamma_split 55
```

- For `EmerGNN`,
```bash
python -W ignore evaluate.py --dataset=S0_finger_55 --n_epoch=40 --epoch_per_test=2 --gpu=0
```

- For `DDI-GPT`, 
```bash
python drugbank/main_drugbank.py --split_strategy cluster
```

## Real Scene
**Real_scene_drugbank** includes DDI data from DrugBank, where drugs are divided into three sequential training-validation-test sets based on their market approval timeline.
