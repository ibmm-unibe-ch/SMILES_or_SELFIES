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

Whole pipeline works as follows:
1. Run assignment of atom types on one of the Molnet task test sets specified (line 310, e.g. delaney) and determine the folder you want to save the output files to (line 336). Then run 
```
python 1_atomtype_assignment.py. 
```
2. Run creation of dictionaries that assign SMILES mainly to clean tokenized SMILES and to assigned atom types. Run 
```
python 2_SMILEStoAtomAssignments.py
```
    This file creates two files: 

        1. **dikt_task.json** that contains as keys the SMILES and as subkeys
        - the index positions in the original array that are atoms ("posToKeep"), 
        - an array of the tokenized version of the SMILES that is devoit of structural tokens ("smi_clean"), 
        - an array that contains the atom types corresponding to the tokenized version in the same order ("atom_types") and 
        - the maximum penalty assigned to this atom assignment as output by parmchk ("max_penalty"). Using the maximum penalty one can filter for the best/most confident atom assignments. Beware that the score returned by parmchk is open-ended and partially ambiguous. 

        2. **assignment_info_task.json** that contains info on failed assignments.

3. Run plotting: This file reads in the previously created dictionaries and info files, gets the embeddings for the task, connects embeddings with atom assignments, clusters the data on elements in a sensible way and plots PCA, UMAP, and LDA for the embeddings labelled to their assigned atom types.
```
python 3_AssignEmbedsPlot.py
```

If you only want to plot PCAs of embeddings of assigned atoms for one of the many Molnet tasks, run (and before that change the folder name)
```
python atomtype_embedding_assignment.py
```
Run atomtype_embeddings22may.py to analyse embeddings of atomtypes for models in specified paths for specified task molecules using PCA, UMAP, and LDA.
This file checks for files with atom types in the specified folder, loads the embeddings for the specified tasks, connects them, and then runs PCA and plots the outcome in several plots.

## Existing Folders (TBC)
```
./assignment_dicts
``` 
contains dictionaries and infos on failed atom assignments on three Molnet test set tasks.

For further info on atom type definitions see:
https://ambermd.org/antechamber/gaff.html#atomtype 


