# this file is to get atom assignments to SMILES from previously created older with mol2-files, atomassignment outputs from antechamber 
# and parmchk2-outputfiles that were created when checking the antechamber output
# Output is a dictionary with SMILES as keys and a dictionary with the following keys: posToKeep, smi_clean, atom_types, max_penalty

from io import StringIO
import os
import numpy as np
import logging
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path
import re
import pandas as pd
from rdkit import Chem
from deepchem.feat import RawFeaturizer
from tokenisation import tokenize_dataset, get_tokenizer
import json

from constants import (
    MOLNET_DIRECTORY,
    TOKENIZER_PATH
)

def load_molnet_test_set(task: str) -> Tuple[List[str], List[int]]:
    """Load MoleculeNet task

    Args:
        task (str): MoleculeNet task to load

    Returns:
        Tuple[List[str], List[int]]: Features, Labels
    """
    task_test = MOLNET_DIRECTORY[task]["load_fn"](
        featurizer=RawFeaturizer(smiles=True), splitter=MOLNET_DIRECTORY[task]["split"]
    )[1][2]
    task_SMILES = task_test.X
    task_labels = task_test.y
    return task_SMILES, task_labels

def clean_acout(ac_out) -> list:
    """Clear hydrogens from atom type 

    Args:
        ac_out (_list_): List of atomtypes from antechamber output

    Returns:
        list: List of antechamber assigned atom types without hydrogens.
    """
    ac_out_noH = list()
    for j in ac_out:
        if not j.startswith('H') and not j.startswith('h'):
            ac_out_noH.append(j)
    #print("before: ", ac_out)
    #print("after: ", ac_out_noH)
    return ac_out_noH

def get_atom_assignment(mol2):
    """Extracting the assignment of atom types to atoms from antechamber output

    Args:
        mol2 (_string_): Name of antechamber-mol2-outputfile

    Returns:
        _list,set_: List of assigned atom types without hydrogens and set of assigned atomtypes
    """
    # extract lines between @<TRIPOS>ATOM and @<TRIPOS>BOND to get atom asss
    with open(mol2) as infile:
        lines = infile.read().splitlines()
    start = [i for i, line in enumerate(
        lines) if line.startswith("@<TRIPOS>ATOM")][0]
    end = [i for i, line in enumerate(
        lines) if line.startswith("@<TRIPOS>BOND")][0]
    extract = "\n".join(lines[start+1:end])
    #print("\n_______________________extraction \n", extract)
    pddf = pd.read_csv(StringIO(extract), header=None, delimiter=r"\s+")
    # extract 5th column with atom_asss
    atoms_assigned_list = pddf.iloc[:, 5].tolist()
    # clean H from atom assignment
    atoms_assigned_list_clean = clean_acout(atoms_assigned_list)
    atoms_assigned_set = set(atoms_assigned_list_clean)
    return atoms_assigned_list_clean, atoms_assigned_set

def check_parmchk2(file):
    """Checking of antechamber-assignment file with parmchk2 if revision is required and getting maximum penalty score from file
        Checking of parmchk2-created file for atomtype assignment, 
        see https://docs.bioexcel.eu/2020_06_09_online_ambertools4cp2k/04-parameters/index.html for more info
            --> if “ATTN: needs revision” is found in file, the atomtype assignment failed

    Args:
        file (_string_): Inputfile name

    Returns:
        _bool_, _int_: True if parmchk2 file is ok, False if it calls for revision of antechamber file
    """
    pen_list = list()
    with open(file) as infile:
        lines = infile.read().splitlines()
        for line in lines:
            if "ATTN: needs revision" in line:
                logging.info(
                    f"Atom assignment failed: parmchk2 calls for revision for file {file}")
                print(
                    "################################ATTENTION######################################")
                print(
                    "#####################Parmchk2: atomtypes need revision#########################")
                return False,None
            if "penalty score" in line:
                penalty_score_str = line.split("penalty score")[1].strip().split("=")[1].strip()
                penalty_score = float(re.search(r"\d+\.?\d*", penalty_score_str).group())
                #print(line)
                #print(penalty_score)
                pen_list.append(penalty_score)
    if (len(pen_list))==0:
        pen_list.append(0.0)
    max_penalty = max(pen_list)   
    return True,max_penalty

def clean_SMILES(SMILES_tok):
    """Cleaning of SMILES tokens input from hydrogens and digits

    Args:
        SMILES_tok (_list_): List of SMILES_tokens for a given SMILES

    Returns:
        _list,list_: Processed SMILES_token list and list of positions in input tokens list that were kept 
        (needed to distinguish which embeddings are relevant)
    """
    SMILES_tok_prep = list()
    struc_toks = r"()=:~1234567890#/\\"
    posToKeep = list()
    pos = 0
    for i in range(len(SMILES_tok)):
        # when it's an H in the SMILES, ignore, cannot deal
        if SMILES_tok[i] != "H" and SMILES_tok[i] != "h" and not SMILES_tok[i].isdigit() and not SMILES_tok[i].isspace():
            if not any(elem in struc_toks for elem in SMILES_tok[i]):
                if SMILES_tok[i] != "-":
                    SMILES_tok_prep.append(SMILES_tok[i])
                    # keep pos where you keep SMILES token
                    posToKeep.append(pos)
        pos += 1
    assert(len(posToKeep) == (len(SMILES_tok_prep))
           ), f"Length of positions-to-keep-array ({len(posToKeep)}) and length of SMILES_tok_prep ({len(SMILES_tok_prep)}) are not the same"
    print("SMILES_tok: ", SMILES_tok)
    print("posToKeep: ", posToKeep)
    print("SMILES_tok_prep: ", SMILES_tok_prep)

    return SMILES_tok_prep, posToKeep

def load_assignments_from_folder(folder, smiles_tokens_dict, task_SMILES):
    """Function to load atom assignments from folder given folder with mol2-files, atomassignment outputs from antechamber 
    and parmchk2-outputfiles that were created when checking the antechamber output

    Args:
        folder (_string_): Name of folder from which to load files
        smiles_arr (_list_): Array of SMILES
        smi_toks (_list_): List of lists that corresponds to smiles_arr and contains the tokens to th corresponding SMILES

    Returns:
        _dict,dict,list,int,list,list_: Many: dikt, dikt_clean, posToKeep_list, filecreation_fail+assignment_fail, failedSmiPos, cleanSmis
    """
    failedSmiPos = list()
    acfailedSMILES = list()
    lengthfail=list()
    parmchkfailedSMILES = list()
    assignment_list = list()
    dikt = dict()
    posToKeep_list = list()
    mol2_files = list()
    assignment_fail = 0
    # get all atom assignment files
    for file in os.listdir(folder):
        if file.endswith(".mol2") and not file.endswith("assigned.mol2"):
            mol2_files.append(file)
    # sort according to numbers
    mol2_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print(mol2_files)
    print("len of mol2files: ",len(mol2_files))

    filecreation_fail = len(smiles_tokens_dict.keys())-(len(mol2_files))
    # I assume all file creations worked, if not this fails.
    assert(len(mol2_files) == (len(smiles_tokens_dict.keys()))
           ), f"Not every SMILES ({len(smiles_tokens_dict.keys())}) has a corresponding file ({len(mol2_files)}) created for it. Needs more checking."
    for mol2 in mol2_files:
        num = int((re.findall(r'\d+', mol2.split('.')[0]))[0])
        print(num)
        parmcheck_file = f"mol_{num}_assigned.frcmod"
        assignment_file = f"mol_{num}_assigned.mol2"
        smi = task_SMILES[num]
        print(f"smi {smi} with tokens {smiles_tokens_dict[smi]}")
        # to check lengths agree, any unknown token will be assumed to have length one, just because it often is "." as in e.g. bbbp dataset
        if(len(smi)!=(sum(1 if s =='<unk>' else len(s) for s in smiles_tokens_dict[smi]))):
            print(f"SMILES and tokenised version do not have same length {smi} with len{len(smi)} to {smiles_tokens_dict[smi]} with len {sum(len(s) for s in smiles_tokens_dict[smi])}")
            dikt[smi] = {"posToKeep": None, "smi_clean": None, "atom_types": None, "max_penalty": None}
            failedSmiPos.append(num)
            assignment_fail += 1
            lengthfail.append(smi)
            continue
        
        print(f"smi {smi} with tokens {smiles_tokens_dict[smi]}")
        smi_clean, posToKeep = clean_SMILES(smiles_tokens_dict[smi])
        print(f"smi: {smi}, smiles_tokens {smiles_tokens_dict[smi]}, smi_clean {smi_clean}")

        print(f"num {num} extracted from mol2 {mol2}")
        # check whether assignment exists
        if os.path.isfile(f"{folder}/{assignment_file}") == True:
            #print("assignment exists")
            # check whether parmcheck exists and is ok, if yes, save output of assignment file
            if os.path.isfile(f"{folder}/{parmcheck_file}") == True:
                is_okay, max_penalty = check_parmchk2(f"{folder}/{parmcheck_file}")
                if is_okay == True:
                    #print(f"parmchk file exists and is ok, max penalty is : {max_penalty}")
                    # then get atom assignments from assignment file
                    atoms_assignment_list, atoms_assignment_set = get_atom_assignment(
                        f"{folder}/{assignment_file}")
                    print(atoms_assignment_list)
                    
                    assignment_list.append(atoms_assignment_list)
                    print(f"({len(atoms_assignment_list)} == {len(smi_clean)})")
                    assert(len(atoms_assignment_list) == len(smi_clean)), f"Length of atom assignment list ({len(atoms_assignment_list)}) and length of cleaned SMILES ({len(smi_clean)}) are not the same"
                    # check atomtypes aassigned to correct atoms as far as possible
                    for str1, str2 in zip(smi_clean, atoms_assignment_list):
                        assert(str1[1] if str1.startswith("[") else str1[0].lower() == str2[0].lower()), f"Atom assignment failed: {str1} != {str2}"
                    
                    dikt[smi] = {"posToKeep": posToKeep, "smi_clean": smi_clean, "atom_types": atoms_assignment_list, "max_penalty": max_penalty}
                    posToKeep_list.append(posToKeep)
                else:
                    dikt[smi] = {"posToKeep": None, "smi_clean": None, "atom_types": None, "max_penalty": None}
                    failedSmiPos.append(num)
                    assignment_fail += 1
                    parmchkfailedSMILES.append(smi)
        else:
            dikt[smi] = {"posToKeep": None, "smi_clean": None, "atom_types": None, "max_penalty": None}
            failedSmiPos.append(num)
            assignment_fail += 1
            acfailedSMILES.append(smi)
    totalfails = filecreation_fail+assignment_fail

    assert(len(dikt.keys()) == (len(smiles_tokens_dict.keys()))
           ), f"Number of keys for SMILES in dictionary ({len(dikt.keys())}) not equal to number of SMILES in original dict minus failures ({len(smiles_tokens_dict.keys())})"
    assert len(posToKeep_list) == len(smiles_tokens_dict.keys())-totalfails, f"Length of list of positions of assigned atoms in SMILES ({len(posToKeep_list)}) and number of SMILES tokens minus failures ({len(smiles_tokens_dict.keys())}) is not the same."
    logging.info(
        f"File creation from SMILES to pdb by obabel failed {filecreation_fail} times out of {len(task_SMILES)}")
    logging.info(
        f"Atom assignment by antechamber failed {assignment_fail} times out of {len(task_SMILES)}")
    assert(len(failedSmiPos) == len(acfailedSMILES)+len(parmchkfailedSMILES)+len(lengthfail)), f"Length of failed SMILES positions ({len(failedSmiPos)}) and  ac failed SMILES ({len(acfailedSMILES)}) + parmcheck failed smiles ({len(parmchkfailedSMILES)}) is not the same."
    logging.info(f"Length of failed SMILES positions ({len(failedSmiPos)}) of these antechamber failed SMILES: ({len(acfailedSMILES)}), parmcheck failed SMILES: ({len(parmchkfailedSMILES)}).")
    logging.info(f"Antechamber failed to assign atom types to the following ({len(acfailedSMILES)}) SMILES: {acfailedSMILES}")
    logging.info(f"The following SMILES did not have parmchk files: {parmchkfailedSMILES}")
    
    return dikt, totalfails, failedSmiPos, posToKeep_list
    
#from preprocessing
def canonize_smiles(input_str: str, remove_identities: bool = True) -> str:
    """Canonize SMILES string

    Args:
        input_str (str): SMILES input string

    Returns:
        str: canonize SMILES string
    """
    mol = Chem.MolFromSmiles(input_str)
    if mol is None:
        return None
    # not sure remove_identities is neccessary for generate mapping, cannot see a difference
    if remove_identities:
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]

    return Chem.MolToSmiles(mol)

def get_tokenized_SMILES(task_SMILES_orig: List[str]):
    """Tokenize SMILES string

    Args:
        input_list of strings (str): List of SMILES input string

    Returns:
        dict: dictionary that links canonize SMILES string
    """

    #if tokenizer_set==False:
    #    #### TEST: TURN ALL SMALL C INZTO CAPITALIZED VERSIONS TO SEE EFFECT ON EMBEDDINGS
    #    task_SMILES = [smiles.replace("c","C") for smiles in task_SMILES_orig]
    #    print(task_SMILES)
    #    #tokenised_smiles = [elem for elem in re.split(PARSING_REGEX,smiles) if elem]
    #    tokenised_smiles = [re.split(PARSING_REGEX,smiles) for smiles in task_SMILES if isinstance(smiles,str)]
    #    print(f"SMILES tokens: {tokenised_smiles}")
    #    tokenised_smiles = [[elem for elem in tokens if elem] for tokens in tokenised_smiles]  # Remove empty tokens
    #    print(f"SMILES tokens: {tokenised_smiles}")
    #    smiles_dict = dict(zip(task_SMILES_orig,tokenised_smiles))
    #else:
    tokenizer = get_tokenizer(TOKENIZER_PATH)
    print(f"tokenizer {tokenizer}")
    smi_toks = tokenize_dataset(tokenizer, task_SMILES, False)
    smi_toks = [smi_tok.split() for smi_tok in smi_toks]
    print(f"SMILES tokens: {smi_toks[0]}")
    smiles_dict = dict(zip(task_SMILES,smi_toks))
    return smiles_dict

def save_assignments_to_file(outfolder, dikt, totalfails, failedSmiPos, posToKeep_list, task):
    
    with open(f'{outfolder}/dikt_{task}.json', 'w') as file:
        json.dump(dikt, file, indent=4)

    print("Dictionary saved to dikt.json")

    # Load the dictionary from the JSON file
    with open(f'{outfolder}/dikt_{task}.json', 'r') as file:
        loaded_dikt = json.load(file)

    print("Dictionary loaded from dikt.json")
    print(loaded_dikt)
    
    # save infos to file
    data = {
    "totalfails": totalfails,
    "failedSmiPos": failedSmiPos,
    "posToKeep_list": posToKeep_list
    }
    
    #df = pd.DataFrame(data)

    # Save to CSV
    #df.to_csv(f'{outfolder}/assignment_info{task}.csv', index=False)
    
    with open(f'{outfolder}/assignment_info_{task}.json', 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {outfolder}assignment_info_{task}.json")
    

    
if __name__ == "__main__":
    print("Start")
    task = "delaney"
    #task = "bace_classification"
    #task="clearance"
    #task="bbbp"
    assert task in list(
        MOLNET_DIRECTORY.keys()
    ), f"{task} not in MOLNET tasks."
    
    # get SMILES from task
    task_SMILES, task_labels = load_molnet_test_set(task)
    print(f"SMILES: {task_SMILES} \n len task_SMILES {task}: {len(task_SMILES)}")
    
    # make sure all task SMILES are the canonical SMILES
    task_SMILES = [canonize_smiles(smiles) for smiles in task_SMILES]
    
    # get tokenized version of dataset, SMILES mapped to tokenised version
    smiles_dict = get_tokenized_SMILES(task_SMILES)
    for key, val in smiles_dict.items():
        print(f"{key}: {val}")
    
    
    folder=f"/data/ifender/SOS_atoms/{task}_mols_bccc0_gaff2_assigned/"
    if task=="bace_classification":
        folder="/home/ifender/SOS/SMILES_or_SELFIES/atomtype_embedding_visualisation/bace_classification_mols_bccc0_gaff2_assigned"
    if task=="delaney":
        folder="/home/ifender/SOS/SMILES_or_SELFIES/atomtype_embedding_visualisation/delaney_mols_bccc0_gaff2_assigned"
    # get assignments and save to file
    dikt, totalfails, failedSmiPos, posToKeep_list = load_assignments_from_folder(folder, smiles_dict, task_SMILES)
    print(f"totalfails: {totalfails}, failedSmiPos: {failedSmiPos}")
    print(f"failed SMILES: {failedSmiPos}")
    print(f"dikt: {dikt}")
        
    # save dikt to file and also totalfails, failedSmiPos, posToKeep_list to info file
    outfolder = "/home/ifender/SOS/SMILES_or_SELFIES/atomtype_embedding_visualisation/assignment_dicts"
    save_assignments_to_file(outfolder, dikt, totalfails, failedSmiPos, posToKeep_list, task)
    
    with open(f"{outfolder}/dikt_{task}.json", 'r') as file:
        loaded_dikt = json.load(file)
    print(loaded_dikt)
    print("DIKT loaded")
    for key,val in loaded_dikt.items():
        print(f"{key}: {val}")
    print(len(loaded_dikt.keys()))
    
