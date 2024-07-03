from io import StringIO
import os
import numpy as np
import logging
from typing import List, Tuple
import umap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path
import re
import pandas as pd
from fairseq_utils import compute_model_output, compute_model_output_RoBERTa, load_dataset, load_BART_model, load_model
from fairseq.data import Dictionary
from deepchem.feat import RawFeaturizer
from tokenisation import tokenize_dataset, get_tokenizer
from constants import SEED
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# requires Python 3.10.14

from constants import (
   # TASK_MODEL_PATH,
    TASK_PATH,
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

def get_tokenized_SMILES(task_SMILES: List[str]):
    tokenizer = get_tokenizer(TOKENIZER_PATH)
    print(f"tokenizer {tokenizer}")
    smi_toks = tokenize_dataset(tokenizer, task_SMILES, False)
    smi_toks = [smi_tok.split() for smi_tok in smi_toks]
    print(f"SMILES tokens: {smi_toks[0]}")
    smiles_dict = dict(zip(task_SMILES,smi_toks))
    return smiles_dict

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

def load_assignments_from_folder(folder, smiles_tokens_dict, task_SMILES):
    """Function to load atom assignments from folder given foder with mol2-files, atomassignment outputs from antechamber 
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
    parmchkfailedSMILES = list()
    assignment_list = list()
    dikt = dict()
    posToKeep_list = list()
    mol2_files = list()
    assignment_fail = 0
    # get all atom assignment files
    for file in os.listdir(folder):
        if file.endswith(".mol2") and not file.endswith("ass.mol2"):
            mol2_files.append(file)
    # sort according to numbers
    mol2_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print(mol2_files)
    #print("len of mol2files: ",len(mol2_files))

    filecreation_fail = len(smiles_tokens_dict.keys())-(len(mol2_files))
    # I assume all file creations worked, if not this fails.
    assert(len(mol2_files) == (len(smiles_tokens_dict.keys()))
           ), f"Not every SMILES ({len(smiles_tokens_dict.keys())}) has a corresponding file ({len(mol2_files)}) created for it. Needs more checking."
    for mol2 in mol2_files:
        num = int((re.findall(r'\d+', mol2.split('.')[0]))[0])
        print(num)
        parmcheck_file = f"mol_{num}_ass.frcmod"
        assignment_file = f"mol_{num}_ass.mol2"
        smi = task_SMILES[num]
        assert(len(smi)==(sum(len(s) for s in smiles_tokens_dict[smi]))), f"SMILES and tokenised version do not have same length {smi} to {smiles_tokens_dict[smi]}"
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
                    
                    dikt[smi] = {"posToKeep": posToKeep, "smi_clean": smi_clean, "atomtypes": atoms_assignment_list, "max_penalty": max_penalty}
                    posToKeep_list.append(posToKeep)
                else:
                    dikt[smi] = {"posToKeep": None, "smi_clean": None, "atomtypes": None, "max_penalty": None}
                    failedSmiPos.append(num)
                    assignment_fail += 1
                    parmchkfailedSMILES.append(smi)
        else:
            dikt[smi] = {"posToKeep": None, "smi_clean": None, "atomtypes": None, "max_penalty": None}
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
    assert(len(failedSmiPos) == len(acfailedSMILES)+len(parmchkfailedSMILES)), f"Length of failed SMILES positions ({len(failedSmiPos)}) and  ac failed SMILES ({len(acfailedSMILES)}) + parmcheck failed smiles ({len(parmchkfailedSMILES)}) is not the same."
    logging.info(f"Length of failed SMILES positions ({len(failedSmiPos)}) of these antechamber failed SMILES: ({len(acfailedSMILES)}), parmcheck failed SMILES: ({len(parmchkfailedSMILES)}).")
    logging.info(f"Antechamber failed to assign atom types to the following ({len(acfailedSMILES)}) SMILES: {acfailedSMILES}")
    logging.info(f"The following SMILES did not have parmchk files: {parmchkfailedSMILES}")
    
    return dikt, totalfails, failedSmiPos, posToKeep_list

def get_embeddings(task: str, specific_model_path: str, data_path: str, cuda: int):
    """Generate the embeddings dict of a task
    Args:
        task (str): Task to find attention of
        cuda (int): CUDA device to use
    Returns:
        Tuple[List[List[float]], np.ndarray]: attention, labels
    """
    task_SMILES, task_labels = load_molnet_test_set(task)

    #data_path = "/data/jgut/SMILES_or_SELFIES/task/delaney/smiles_atom_isomers"
    model = load_model(specific_model_path, data_path, cuda)
    #print("model loaded")
    model.zero_grad()
    data_path = data_path / "input0" / "test"
    # True for classification, false for regression
    dataset = load_dataset(data_path, True)
    source_dictionary = Dictionary.load(str(data_path.parent / "dict.txt"))

    assert len(task_SMILES) == len(
        dataset
    ), f"Real and filtered dataset {task} do not have same length."

    #text = [canonize_smile(smile) for smile in task_SMILES]
    text = [smile for smile in task_SMILES]
    embeds= []
    tokenizer = None
    if "bart" in str(specific_model_path):
        embeds.append(
            compute_model_output(
                dataset,
                model,
                text, #this is very important to be in same order as task_SMILES which it is
                source_dictionary,
                False,
                False,
                True,  # true for embeddings
                True,  # true for eos_embeddings
                tokenizer,
            )[2]
        )
    if "roberta" in str(specific_model_path):
        embeds.append(
            compute_model_output_RoBERTa(
                dataset,
                model,
                text,
                source_dictionary,
                False,
                False,
                True,  # true for embeddings
                True,  # true for eos_embeddings
                tokenizer,
            )[2]
        )
   # print("attention encodings",len(attention_encodings[0]))
   # print(len(attention_encodings))
    output = list(zip(*embeds))
    labels = np.array(task_labels).transpose()[0]
    # print("labels",labels)
    # print(len(labels))
    return embeds

def get_embeddings_from_model(task, TASK_MODEL_PATH, traintype, model):
    # ----------------------specific model paths for Delaney for BART and RoBERTa-------------------------
    # path to finetuned models
    if traintype=="finetuned":
        if model=="BART":
            # path for BART  
            specific_model_path = (
            TASK_MODEL_PATH
            / task
            / "smiles_atom_isomers_bart"
            / "1e-05_0.2_seed_0" 
            / "checkpoint_best.pt"
            )
        else:
            #path for RoBERTa
            specific_model_path = (
                TASK_MODEL_PATH
                / task
                / "smiles_atom_isomers_roberta"
                / "1e-05_0.2_seed_0" 
                / "checkpoint_best.pt"
            )
    # ----------------------specific model paths for pretrained models of BART and RoBERTa-------------------------
    elif traintype=="pretrained":
        if model=="BART":
            # path for BART   
            specific_model_path = (
                TASK_MODEL_PATH
                / "smiles_atom_isomers_bart"
                / "checkpoint_last.pt"
            ) 
        else:
            #path for RoBERTa
            specific_model_path = (
            TASK_MODEL_PATH
            / "smiles_atom_isomers_roberta"
            / "checkpoint_last.pt"
            )
    print("specific model path: ",specific_model_path)
    data_path = TASK_PATH / task / "smiles_atom_isomers"
    
    embeds = []
    embeds = get_embeddings(task, specific_model_path, data_path, False) #works for BART model with newest version of fairseq on github, see fairseq_git.yaml file
    #print("got the embeddings")
    return embeds

def check_lengths(smi_toks, embeds):
    """Check that number of tokens corresponds to number of embeddings per SMILES, otherwise sth went wrong

    Args:
        smi_toks (_list[string]_): SMILES tokens for a SMILES
        embeds (_list[float]_): Embeddings
    """
    samenums = 0
    diffnums = 0
    smismaller = 0
    new_embs = list()
    for smi, embs in zip(smi_toks, embeds[0]):
        if len(smi) == len(embs):
            samenums += 1
            new_embs.append(embs)
        else:
            print(f"smilen: {len(smi)} emblen: {len(embs)}")
            print(f"{smi} and len diff {len(smi)-len(embs)}")
            diffnums += 1
            if len(smi) < len(embs):
                smismaller += 1
    if diffnums == 0:
        return True
    else:
        print(
            f"samenums: {samenums} and diffnums: {diffnums} of which smiles have smaller length: {smismaller}")
        perc = (diffnums/(diffnums+samenums))*100
        print(
            "percentage of embeddings not correct compared to smiles: {:.2f}".format(perc))
        return False

def get_clean_embeds(embeds, dikt, creation_assignment_fail, task_SMILES):
    """Clean embeddings of embeddings that encode for digits, hydrogens, or structural tokens

    Args:
        embeds (_List[List[float]_): Embeddings of a SMILES
        failedSmiPos (_list_): Positions of SMILES in list where no file and/or assignment could be generated
        posToKeep_list (_list_): List of positions in a SMILES according to tokens that need to be kept (not digits, hydrogens, or structural tokens)

    Returns:
        _list[float]_: Embeddings that do not encode hydrogens, digits, or structural tokens, but only atoms
    """
    posToKeep_list = [value["posToKeep"] for value in dikt.values() if value["posToKeep"] is not None ]
    #only keep embeddings for SMILES where atoms could be assigned to types
    embeds_clean = list()
    for smi, emb in zip(task_SMILES, embeds[0]):
        posToKeep = dikt[smi]["posToKeep"]
        if posToKeep is not None:
            embeds_clean.append(emb)
            dikt[smi]["orig_embedding"]=emb
        else:
            dikt[smi]["orig_embedding"]=None
    
    logging.info(
        f"Length embeddings before removal: {len(embeds[0])}, after removal where atom assignment failed: {len(embeds_clean)}")
    assert creation_assignment_fail == (len(
        embeds[0])-len(embeds_clean)), f"Assignment fails ({creation_assignment_fail}) and number of deleted embeddings do not agree ({(len(embeds[0])-len(embeds_clean))})."

    embeds_cleaner = []
    assert len(embeds_clean) == (len([item for item in posToKeep_list if item is not None])
                                 ), f"Not the same amount of embeddings as assigned SMILES. {len(embeds_clean)} embeddings vs. {len([item for item in posToKeep_list if item is not None])} SMILES with positions"
    # only keep embeddings that belong to atoms
    for SMILES in task_SMILES:
        poslist = dikt[SMILES]["posToKeep"]
        emb_clean = dikt[SMILES]["orig_embedding"]

        if poslist is not None:
            newembsforsmi = []
            newembsforsmi = [emb_clean[pos] for pos in poslist]
            embeds_cleaner.append(newembsforsmi)
            dikt[SMILES]["clean_embedding"]=newembsforsmi  
        else:
            dikt[SMILES]["clean_embedding"]=None   

    # sanity check that the lengths agree
    for smiemb, pos_list in zip(embeds_cleaner, posToKeep_list):
        assert len(smiemb) == len(
            pos_list), "Final selected embeddings for assigned atoms do not have same length as list of assigned atoms."
        #print(len(smiemb), pos_list)
        
    # sanity check that length of assigned atoms map to length of clean embeddings
    for SMILES in task_SMILES:
        smi_clean=dikt[SMILES]["smi_clean"]
        emb_clean = dikt[SMILES]["clean_embedding"]
        if dikt[SMILES]["posToKeep"] is not None:
            assert len(smi_clean) == len(
                emb_clean), "SMILES and embeddings do not have same length."
            for sm, em in zip(smi_clean,emb_clean):
                #print(f"sm {sm} em {em[1]}")
                assert(sm==em[1]), f"Atom assignment failed: {sm} != {em[1]}"
    logging.info("Cleaning embeddings finished, all checks passed")
    return embeds_cleaner

if __name__ == "__main__":

    # get SMILES from task
    task = "delaney"
    #task = "bace_classification"
    assert task in list(
        MOLNET_DIRECTORY.keys()
    ), f"{task} not in MOLNET tasks."
    
    # get SMILES from task
    task_SMILES, task_labels = load_molnet_test_set(task)
    print(f"SMILES: {task_SMILES} \n len task_SMILES delaney: {len(task_SMILES)}")

    # get tokenized version of dataset, SMILES mapped to tokenised version
    smiles_dict = get_tokenized_SMILES(task_SMILES)
    
    # get atom assignments from folder that contains antechamber atom assignments and parmchk files
    #smiToAtomAssign_dict, smiToAtomAssign_dict_clean, posToKeep_list, creation_assignment_fail, failedSmiPos, cleanSmis = load_assignments_from_folder(
        #"./delaney_mols_bccc0_gaff2_assigned", smiles_dict, task_SMILES)
    dikt, totalfails, failedSmiPos, posToKeep_list = load_assignments_from_folder("./delaney_mols_bccc0_gaff2_assigned", smiles_dict, task_SMILES)
    
    #get embeddings from model
    finetuned_TASK_MODEL_PATH = Path("/data2/jgut/SoS_models")
    pretrained_TASK_MODEL_PATH = Path("/data/jgut/SMILES_or_SELFIES/prediction_models")
    model = "BART"
    traintype = "finetuned"
    embeds = get_embeddings_from_model(task, finetuned_TASK_MODEL_PATH, traintype, model)
    
    # some sanity checks on embeddings per SMILES
    assert (len(dikt.keys())) == (len(
        embeds[0])), f"Number of SMILES and embeddings do not agree. Number of SMILES: {len(dikt.keys())} of which {totalfails} failures and Number of embeddings: {len(embeds[0])}"
    print(f"Number of SMILES: {len(dikt.keys())} with {totalfails} failures and Number of embeddings: {len(embeds[0])}")
    #check that every token has an embedding
    assert check_lengths(
        smiles_dict.values(), embeds), "Length of SMILES_tokens and embeddings do not agree."
    #print("embeddings passed length checks")

    #get rid of embeddings that encode for digits or hydrogens
    embeds_clean = get_clean_embeds(embeds, dikt, totalfails, task_SMILES)
    #assert check_lengths()
