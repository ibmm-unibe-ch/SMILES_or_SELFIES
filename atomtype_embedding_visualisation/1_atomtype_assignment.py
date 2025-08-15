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

from constants import (
    MOLNET_DIRECTORY,
    TOKENIZER_PATH
)
#from preprocessing and also exists in SMILES_to_SELFIES_mapping
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

def smiles_to_file(smiles, no, ftype,filepath):
    """Execution of babel - generation  of file from SMILES

    Args:
        smiles (_string_): SMILES
        no (_int_): Number that goes in input file name
        ftype (_string_): pdb or mol2

    Returns:
        _string_: Resulting filename or None if execution fails
    """
    if ftype == "pdb":
        os.system(f'obabel -:"{smiles}" -o pdb -O {filepath}mol_{no}.pdb')
        return f"{filepath}mol_{no}.pdb"
    elif ftype == "mol2":
        os.system(
            f'obabel -:"{smiles}" -o mol2 -O {filepath}mol_{no}.mol2 --gen3d')
        return f"{filepath}mol_{no}.mol2"
    else:
        print("Execution of obabel failed, no file could be created. Wrong filetype given. Output filetype needs to be pdb or mol2.")
        return None


def exec_antechamber(inputfile, ftype):
    """Execution of antechamber - atomtype assignment of atoms from file using gaff2 forcefield, 
    see: https://docs.bioexcel.eu/2020_06_09_online_ambertools4cp2k/04-parameters/index.html for more info
     command: 'antechamber -i GWS.H.pdb -fi pdb -o GWS.mol2 -fo mol2 -c bcc -nc 0 -at gaff2'
                -c bcc --> AM1-BCC2 charge method
                -nc 0 --> net charge of molecule 0
                -fo mol2 --> output file format mol2
                -at gaff2 --> forcefield gaff2
    Args:
        inputfile (_string_): Name of the inputfile
        ftype (_string_): Filtetype of the input: pdb or mol2

    Returns:
        _string_: Name of resulting antechamber-file or None if assignment failed.
    """
    # 'antechamber -i GWS.H.pdb -fi pdb -o GWS.mol2 -fo mol2 -c bcc -nc 0 -at gaff2'
    inputfile_noex = os.path.splitext(inputfile)[0]
    outfile = f"{inputfile_noex}_assigned.mol2"
    if ftype == "pdb":
        os.system(
            f"antechamber -i {inputfile} -fi pdb -o {outfile} -fo mol2 -c bcc -nc 0 -at gaff2")
    elif ftype == "mol2":
        os.system(
            f"antechamber -i {inputfile} -fi mol2 -o {outfile} -fo mol2 -c bcc -nc 0 -at gaff2")
    else:
        print("Execution of antechamber failed. Wrong filetype given. Filetype needs to be pdb or mol2.")
        return None
    return outfile


def check_parmchk2(file):
    """Checking of parmchk2-created file for atomtype assignment, 
    see https://docs.bioexcel.eu/2020_06_09_online_ambertools4cp2k/04-parameters/index.html for more info
        --> if “ATTN: needs revision” is found in file, the atomtype assignment failed

    Args:
        file (_string_): Inputfile name

    Returns:
        _bool_: True if parmchk2 file is ok, False if it calls for revision of antechamber file
    """
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
                return False
        return True


def run_parmchk2(ac_outfile):
    """Running of parmchk2 to check atomtype assignment of files with antechamber, 
    see https://docs.bioexcel.eu/2020_06_09_online_ambertools4cp2k/04-parameters/index.html for more info
      command: 'parmchk2 -i GWS.mol2 -f mol2 -o GWS.frcmod -s gaff2'
       --> looking for missing parameters in the atnechamber atomtype assigned mol2 file

    Args:
        ac_outfile (_string_): Name of antechamber-file that contains assigned atoms

    Returns:
        _bool_: True if parmchk2 file is ok, False if it calls for revision of antechamber file
    """
    acout_noex = os.path.splitext(ac_outfile)[0]
    print("acout_noex", acout_noex)
    #run parmchk2 in terminal
    os.system(
        f"parmchk2 -i {ac_outfile} -f mol2 -o {acout_noex}.frcmod -s gaff2")
    if os.path.isfile(f"{acout_noex}.frcmod"):
        return check_parmchk2(f"{acout_noex}.frcmod")
    else:
        return False


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
        _list,set_: List of assigned atom types without cleared of hydrogens and set of assigned atomtypes
    """
    # extract lines between @<TRIPOS>ATOM and @<TRIPOS>BOND to get atom asss
    with open(mol2) as infile:
        lines = infile.read().splitlines()
    start = [i for i, line in enumerate(
        lines) if line.startswith("@<TRIPOS>ATOM")][0]
    end = [i for i, line in enumerate(
        lines) if line.startswith("@<TRIPOS>BOND")][0]
    extract = "\n".join(lines[start+1:end])
    print("\n_______________________extraction \n", extract)
    pddf = pd.read_csv(StringIO(extract), header=None, delimiter=r"\s+")
    # extract 5th column with atom_asss
    atoms_assigned_list = pddf.iloc[:, 5].tolist()
    # clean H from atom assignment
    atoms_assigned_list_clean = clean_acout(atoms_assigned_list)
    atoms_assigned_set = set(atoms_assigned_list_clean)
    return atoms_assigned_list_clean, atoms_assigned_set


def clean_SMILES(SMILES_tok):
    """Cleaning of SMILES tokens input from hydrogens and digits

    Args:
        SMILES_tok (_list_): List of SMILES_tokens for a given SMILES

    Returns:
        _list,list_: Processed SMILES_token list and list of positions in input tokens list that were kept 
        (needed to distinguish which embeddings are relevant)
    """
    SMILES_tok_prep = list()
    struc_toks = r"()=:~1234567890#"
    posToKeep = list()
    pos = 0
    for i in range(len(SMILES_tok)):
        # when it's an H in the SMILES, ignore, cannot deal
        if SMILES_tok[i] != "H" and SMILES_tok[i] != "h" and not SMILES_tok[i].isdigit() and not SMILES_tok[i].isspace():
            if any(elem in struc_toks for elem in SMILES_tok[i]) == False:
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

def get_atom_assignments(smiles_arr, smi_toks, filepath):
    """Getting the atom assignments

    Args:
        smiles_arr (_list_): Array of SMILES
        smi_toks (_list_): List of lists that corresponds to smiles_arr and contains the tokens to th corresponding SMILES

    Returns:
         _dict,dict,list,int,list,list_: Many: dikt, dikt_clean, posToKeep_list, filecreation_fail+assignment_fail, failedSmiPos, cleanSmis
    """
    no = 0
    assignment_list = list()
    dikt = dict()
    dikt_clean = dict()
    posToKeep_list = list()
    filecreation_fail = 0
    assignment_fail = 0
    smi_num = 0
    failedSmiPos = list()
    cleanSmis = list()
    for smi, smi_tok in zip(smiles_arr, smi_toks):
        # print statements only to structure obabel and antechamber output
        print("##############################################################################################################")
        print(f"SMILES: {smi}")
        # clean SMILES of hydrogens and digits
        smi_clean, posToKeep = clean_SMILES(smi_tok)
        cleanSmis.append(smi_clean)
        #print(f"smi_tok turns to smi_clean: {smi_tok}  --->  {smi_clean}")
        # create mol2-file from SMILES with obabel
        smi_fi = smiles_to_file(smi, no, "mol2", filepath)
        # check whether the file was created, else keep track of failed file creation
        if os.path.isfile(smi_fi) == True:
            print("Successful conversion of SMILES to file")
            # execute antechamber on file
            smi_ac = exec_antechamber(smi_fi, "mol2")
            # check whether antechamber-file-creation worked, else keep track of failed file creation
            if os.path.isfile(smi_ac) == True:
                # if smi_ac was generated check if with parmchk2, returns True if output is ok
                if True == run_parmchk2(smi_ac):
                    # get antechamber assignment (without hydrogens)
                    atoms_assignment_list, atoms_assignment_set = get_atom_assignment(
                        smi_ac)
                    assignment_list.append(atoms_assignment_list)
                    dikt[smi] = (posToKeep, smi_clean, atoms_assignment_list)
                    dikt_clean[smi] = (posToKeep, smi_clean,
                                       atoms_assignment_list)
                    posToKeep_list.append(posToKeep)
                else:
                    assignment_fail += 1
                    dikt[smi] = (None, None, None)
                    failedSmiPos.append(smi_num)
            else:
                assignment_fail += 1
                dikt[smi] = (None, None, None)
                failedSmiPos.append(smi_num)
        else:
            filecreation_fail += 1
            dikt[smi] = (None, None, None)
            failedSmiPos.append(smi_num)
        no += 1
        smi_num += 1
    assert(len(dikt.keys()) == (len(smiles_arr)))
    assert(len(dikt.keys()) == (len(smi_toks)))
    assert len(posToKeep_list) == len(dikt_clean.keys(
    )), f"Length of list of positions of assigned atoms in SMILES ({len(posToKeep_list)}) and number of SMILES ({len(posToKeep_list)}) is not the same."
    logging.info(
        f"File creation from SMILES to pdb by obabel failed {filecreation_fail} times out of {len(rndm_smiles)}")
    logging.info(
        f"Atom assignment by antechamber failed {assignment_fail} times out of {len(rndm_smiles)}")
    return dikt, dikt_clean, posToKeep_list, filecreation_fail+assignment_fail, failedSmiPos, cleanSmis



if __name__ == "__main__":
    ############################### get SMILES from Task ###################################################   
    #task = "delaney"
    #task = "bace_classification"
    #task="bbbp"
    #task="clearance"
    task="lipo"
    assert task in list(
        MOLNET_DIRECTORY.keys()
    ), f"{task} not in MOLNET tasks."
    
    task_SMILES, task_labels = load_molnet_test_set(task)
    task_SMILES = [canonize_smiles(smiles) for smiles in task_SMILES]
    print(
        f"SMILES: {task_SMILES} \n len task_SMILES {task}: {len(task_SMILES)}")
    #print(f"task labels",task_labels)
    rndm_smiles = task_SMILES
    print(f"first smiles {task_SMILES[0]} and length {len(task_SMILES[0])}")
    print(f"{task} reading done")
    
    ###############################get tokenized version of dataset ########################################
    # get tokenized version of dataset
    tokenizer = get_tokenizer(TOKENIZER_PATH)
    print(f"tokenizer {tokenizer}")
    smi_toks = tokenize_dataset(tokenizer, task_SMILES, False)
    print("whole SMILES tokenized: ",smi_toks[0])
    smi_toks = [smi_tok.split() for smi_tok in smi_toks]
    print(f"SMILES tokens after splitting tokens into single strings: {smi_toks[0]}")


    ############################## get atomassignments for task test set using antechamber and parmchk2 (not here: OR from previous antechamber assignment with antechamber and parmchk2) ########################################
        # get atom assignments from SMILES
    filepath = f"/data/ifender/SOS_atoms/{task}_mols_bccc0_gaff2_assigned/"
    #Check if the directory exists, and create it if it doesn't
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    smiToAtomAssign_dict, smiToAtomAssign_dict_clean, posToKeep_list, creation_assignment_fail, failedSmiPos, cleanSmis = get_atom_assignments(task_SMILES,smi_toks,filepath)