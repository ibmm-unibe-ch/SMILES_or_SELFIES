# Attention on atoms visualized

## Set-Up
Create conda env with
```
conda env create --file environment.yml
```

Activate conda env with 

```
conda activate attentionviz2
```

## Visualize attention for SMILES
Chose SMILES and visualize attention per token by running AttentionVisualised_2.ipynb


### Old info:
Activate conda env
```
conda activate attentionviz
```

Pip install remaining installations with
```
python ./install_pip_fromyml.py
```
Extra installs left:
```
conda install rdkit
conda install ipykernel
conda install numpy
pip install matplotlib
pip install deepchem
```