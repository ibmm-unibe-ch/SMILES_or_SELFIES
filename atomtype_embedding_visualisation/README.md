# Visualising embeddings of atomtypes as assigned by Antechamber of SMILES or SELFIES trained models
Atoms of SMILES of a dataset of one of the MOLECULENet datasets (e.g. Delaney) are assigned atomtypes using Antechamber.
The embeddings of these atoms are extracted from the model and analysed with PCA or UMAP, different atomtypes are plotted together.

## Set-Up 
Use the fairseq_git.yml to create a conda environment.
```
conda env create --file fairseq_git.yml
```
Pip installations from .yml will still most likely fail and you will need to install them by hand, e.g. by extracting pip-packages from yaml-file and installing them by running subprocesses.

```
# Load the fairseq_git.yml file
with open('fairseq_git.yml', 'r') as file:
    env = yaml.safe_load(file)

# Extract the pip packages
pip_packages = [pkg for dep in env['dependencies'] if isinstance(dep, dict) for pkg in dep.get('pip', [])]

# Uninstall the pip package (just in case)
subprocess.run(['pip', 'uninstall', '-y'] + pip_packages)

# Reinstall the pip packages
subprocess.run(['pip', 'install'] + pip_packages)
```
Then activate conda environment with
```
conda activate fairseq_git
```

## Run analysis
Run atomtype_embeddings.py.

Understanding atom type definitions:
https://ambermd.org/antechamber/gaff.html#atomtype 


