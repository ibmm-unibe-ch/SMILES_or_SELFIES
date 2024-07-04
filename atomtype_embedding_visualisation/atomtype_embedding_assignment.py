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
from itertools import chain

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

def get_embeddings_from_model(task, TASK_MODEL_PATH, traintype, model, smiles_dict):
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
    assert check_lengths(smiles_dict.values(), embeds), "Length of SMILES_tokens and embeddings do not agree."
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
    # some sanity checks on embeddings per SMILES
    assert (len(dikt.keys())) == (len(
        embeds[0])), f"Number of SMILES and embeddings do not agree. Number of SMILES: {len(dikt.keys())} of which {totalfails} failures and Number of embeddings: {len(embeds[0])}"
    print(f"Number of SMILES: {len(dikt.keys())} with {totalfails} failures and Number of embeddings: {len(embeds[0])}")
    
    #check that every token has an embedding
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

def map_embeddings_to_atomtypes(dikt,task_SMILES):
    for SMILES in task_SMILES:
        if dikt[SMILES]["posToKeep"] is not None:
            atomtype_to_embedding = {}
            atom_types = dikt[SMILES]['atom_types']
            embeddings = dikt[SMILES]['clean_embedding']
            type_to_emb_dict = dict()
            for atom_type, embedding in zip(atom_types, embeddings):
                atomtype_to_embedding.setdefault(atom_type, []).append(embedding)
                type_to_emb_dict[atom_type] = embedding
                assert(atom_type.lower() if atom_type.lower() =='cl' else atom_type[0].lower()==(embedding[1][1].lower() if embedding[1].startswith("[") else embedding[1]).lower()), f"Atom assignment failed: {atom_type} != {embedding[1]}"
            dikt[SMILES]["atomtype_to_embedding"] = type_to_emb_dict
        else:
            dikt[SMILES]["atomtype_to_embedding"]=None
    logging.info("Embeddings mapped to atom types, all checks passed")
    


def colorstoatomtypesbyelement(atomtoelems_dict):
    """Generating a dictionary of colors given a dictionary that maps atomtypes to elements

    Args:
        atomtoelems_dict (_dict_): Dictionary that maps atom types to elements

    Returns:
        _dict,: Dictionary that maps atom types to colors
    """
    # https://sashamaps.net/docs/resources/20-colors/ #95% accessible only, subject to change, no white
    colors_sash = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
                   '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000']
   
    colordict = dict()
    for key in atomtoelems_dict.keys():
        atypes = atomtoelems_dict[key]
        keycoldict=dict()
        for at, col in zip(atypes, colors_sash[0:len(atypes)]):
            keycoldict[at]=col    
        colordict[key]=keycoldict 
    print(colordict.items())
    
    # now instead for each element, get colors for a combination of atomtypes
    # p f cl o s
    key='p f cl o s'
    pfclos_types = atomtoelems_dict['p']+atomtoelems_dict['f']+atomtoelems_dict['cl']+atomtoelems_dict['o']+atomtoelems_dict['s']
    keycoldicti=dict()
    for at, col in zip(pfclos_types, colors_sash[0:len(pfclos_types)]):
        keycoldicti[at]=col
    colordict[key]=keycoldicti 
    # c o
    key='c o'
    pfclos_types = atomtoelems_dict['c']+atomtoelems_dict['o']
    keycoldicti=dict()
    for at, col in zip(pfclos_types, colors_sash[0:len(pfclos_types)]):
        keycoldicti[at]=col
    colordict[key]=keycoldicti 
    print(colordict.keys())
    print(colordict.items())
    return colordict

def create_elementsubsets(atomtype_set):
    """Creation of element subsets according to alphabet

    Args:
        big_set (_set_): Set of atom types
    Returns:
        _list,dict[string][list[float],list[string]]_: List of keys (elements), dictionary that contains embeddings and their atomtypes sorted by element
    """
    atomtype_set=sorted(atomtype_set)
    element_dict = dict()
    elements = list()
    ctr=0
    last_firstval = ''
    for atype in atomtype_set:
        if ctr==0:
            last_firstval = atype[0]
        if not atype.startswith('cl') and atype not in element_dict.items() and atype[0]==last_firstval:
            #print(elements)
            elements.append(atype)
            element_dict[last_firstval] = elements
        elif last_firstval != atype[0] and atype != 'cl':
            element_dict[last_firstval] = elements
            elements = list()
            elements.append(atype)
            last_firstval = atype[0]
        ctr+=1
    element_dict['cl']=['cl']
    return element_dict

def build_legend(data):
    """
    Build a legend for matplotlib plt from dict
    """
    legend_elements = []
    for key in data:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=key,
                                      markerfacecolor=data[key], markersize=9))
    return legend_elements

def plot_umap(embeddings, labels, colours_dict, save_path, min_dist=0.1, n_neighbors=15, alpha=0.2):
    """Performing UMAP and plotting it

    Args:
        embeddings (_list[float]_): Embeddings of one element or a subgroup
        labels (_list[string]_): List of assigned atom types
        colours_dict (_dict[string][int]_): Dictionary of colors linking atomtypes to colors
        set_list (_set(string)_): Set of atomtypes
        save_path (_string_): Path where to save plot
        min_dist (float, optional): Minimum distance for UMAP. Defaults to 0.1.
        n_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 15.
        alpha (float, optional): Level of opacity. Defaults to 0.2.
    """
    logging.info("Started plotting UMAP")
    os.makedirs(save_path.parent, exist_ok=True)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=SEED + 6539
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    fig, ax = plt.subplots(1)
    ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], marker='.', alpha=alpha, c=[
               colours_dict[x] for x in labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(1.13, 0.5), fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=17)
    ax.set_xlabel("UMAP 1", fontsize=17)
    ax.set_title("UMAP - Embeddings resp. to atom types", fontsize=21)
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(length=8, width=3, labelsize=15)
    fig.savefig(save_path, format="svg", bbox_inches='tight', transparent=True)
    fig.clf()


def plot_pca(embeddings, labels, colours_dict, save_path, alpha=0.2):
    """Performing PCA and plotting it

    Args:
        embeddings (_list[float]_): Embeddings of one element or a subgroup
        labels (_list[string]_): List of assigned atom types
        colours_dict (_dict[string][int]_): Dictionary of colors linking atomtypes to colors
        save_path (_string_): Path where to save plot
        alpha (float, optional): Level of opacity. Defaults to 0.2.
    """
    logging.info("Started plotting PCA")
    os.makedirs(save_path.parent, exist_ok=True)
    pca = PCA(n_components=2, random_state=SEED + 6541)
    pca_embeddings = pca.fit_transform(embeddings)
    logging.info(
        f"{save_path} has the explained variance of {pca.explained_variance_ratio_}"
    )
    explained_variance_percentages = [f"{var:.2%}" for var in pca.explained_variance_ratio_]  # Format as percentages
    fig, ax = plt.subplots(1)
    ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], marker='.', alpha=alpha, c=[
               colours_dict[x] for x in labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(1.13, 0.5), fontsize=8)
    ax.set_ylabel(f"PCA 2, var {explained_variance_percentages[1]}", fontsize=17)
    ax.set_xlabel(f"PCA 1, var {explained_variance_percentages[0]}", fontsize=17)
    ax.set_title("PCA - Embeddings resp. atom types", fontsize=21)
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(length=8, width=3, labelsize=15)
    fig.savefig(save_path, format="svg", bbox_inches='tight', transparent=True)
    fig.clf()

def plot_lda(embeddings, labels, colours_dict, save_path, alpha=0.2):
    """Performing Linear Discriminant Analysis and plotting it

    Args:
        embeddings (_list[float]_): Embeddings of one element or a subgroup
        labels (_list[string]_): List of assigned atom types
        colours_dict (_dict[string][int]_): Dictionary of colors linking atomtypes to colors
        save_path (_string_): Path where to save plot
        alpha (float, optional): Level of opacity. Defaults to 0.2.
    """
    logging.info("Started plotting LDA")
    os.makedirs(save_path.parent, exist_ok=True)
    lda = LDA(n_components=2)
    lda_embeddings = lda.fit_transform(embeddings,labels)
    #logging.info(
    #    f"{save_path} has the explained variance of {pca.explained_variance_ratio_}"
    #)
    fig, ax = plt.subplots(1)
    ax.scatter(lda_embeddings[:, 0], lda_embeddings[:, 1], marker='.', alpha=alpha, c=[
               colours_dict[x] for x in labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(1.13, 0.5), fontsize=8)
    ax.set_ylabel("LDA 2", fontsize=17)
    ax.set_xlabel("LDA 1", fontsize=17)
    ax.set_title("LDA - Embeddings resp. atom types", fontsize=21)
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(length=8, width=3, labelsize=15)
    fig.savefig(f"{save_path}.svg", format="svg", bbox_inches='tight', transparent=True)
    fig.clf()
    
    # same but random labels
    lda = LDA(n_components=2)
    random_labels=labels.copy()
    np.random.shuffle(random_labels)
    lda_embeddings = lda.fit_transform(embeddings,random_labels)
    #logging.info(
    #    f"{save_path} has the explained variance of {pca.explained_variance_ratio_}"
    #)
    fig, ax = plt.subplots(1)
    ax.scatter(lda_embeddings[:, 0], lda_embeddings[:, 1], marker='.', alpha=alpha, c=[
               colours_dict[x] for x in random_labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(1.13, 0.5), fontsize=8)
    ax.set_ylabel("LDA 2", fontsize=17)
    ax.set_xlabel("LDA 1", fontsize=17)
    ax.set_title("LDA random - Embeddings resp. atom types", fontsize=21)
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(length=8, width=3, labelsize=15)
    fig.savefig(f"{save_path}_random.svg", format="svg", bbox_inches='tight', transparent=True)
    fig.clf()


def plot_umap_pca_lda(p_f_cl_list_embs, p_f_cl_list_assigs, namestring, save_path_prefix, atomtype2color, min_dist, n_neighbors, alpha):
    #create paths on what to name the plots
    pathway = Path(str(save_path_prefix) +
                   f"umap_{min_dist}_{n_neighbors}_{namestring}.svg")
    pathway_pca = Path(str(save_path_prefix) + f"pca_{namestring}.svg")
    pathway_lda = Path(str(save_path_prefix) + f"lda_{namestring}")
    
    # plot UMAP
    #plot_umap(p_f_cl_list_embs, p_f_cl_list_assigs, atomtype2color, pathway, min_dist, n_neighbors, alpha)
    # plot PCA
    plot_pca(p_f_cl_list_embs, p_f_cl_list_assigs,
             atomtype2color[namestring], pathway_pca, alpha)
    # plot LDA
    plot_lda(p_f_cl_list_embs, p_f_cl_list_assigs,
             atomtype2color[namestring], pathway_lda, alpha)

def create_plotsperelem(dikt, colordict, penalty_threshold, min_dist, n_neighbors, alpha, save_path_prefix):
    """Create plot per element and for all element subsets

    Args:
        dikt (_dict_): Dictionary of atom mappings etc
        colordict (_dict[string][dict[string],[color]]): Dictionary that maps atom types to colors
        penalty_threshold (_float_): Threshold for max penalty score
        min_dist (_float_): Number of min dist to use in UMAP
        n_neighbors (_int_): Number of neighbors to use in UMAP
        alpha (_int_): Level of opacity
        save_path_prefix (_string_): Path prefix where to save output plot
    """
    print(colordict.keys())
    # Assuming 'dikt' is your dictionary and each value has a 'penalty_score' key
    filtered_dict = {smiles: info for smiles, info in dikt.items() if info['max_penalty'] is not None and info['max_penalty'] < penalty_threshold}
    #print(filtered_dict.items())
    for key, value in filtered_dict.items():
        print(value['max_penalty'])
    
    atomtype_to_embedding_dicts = [value['atomtype_to_embedding'] for value in filtered_dict.values() if 'atomtype_to_embedding' in value and value['atomtype_to_embedding'] is not None]
    
    # sort embeddings according to atomtype, I checked it visually and the mapping works
    embeddings_by_atomtype = {}  # Dictionary to hold lists of embeddings for each atom type

    for atom_type_dict in atomtype_to_embedding_dicts:
        # go through single dictionary
        for atom_type, embeddings in atom_type_dict.items():
            print(f"atomtype {atom_type} embeddings {embeddings[1]}")
            if atom_type not in embeddings_by_atomtype:
                embeddings_by_atomtype[atom_type] = []
            # extend the list of embeddings for this atom type(, but only by the embedding not the attached token)
            embeddings_by_atomtype[atom_type].append(embeddings[0])
            print(len(embeddings[0]))
    print(embeddings_by_atomtype.keys())
    
    # sort dictionary that is mapping embeddings to atomtypes to elements so that e.g. all carbon atom types can be accessed at once in one list
    atom_types_repeated = []
    embeddings_list = []
    atomtype_embedding_perelem_dict = dict()
    ctr = 0
    for key in colordict.keys():
        print(f"key {key}")
        for atype in colordict[key]:
            print(atype) 
            if atype in embeddings_by_atomtype.keys():
                embsofatype = embeddings_by_atomtype[atype]
                atypes = [atype] * len(embeddings_by_atomtype[atype])
                assert len(embsofatype) == len(atypes), "Length of embeddings and atom types do not match."
                if key not in atomtype_embedding_perelem_dict:
                    atomtype_embedding_perelem_dict[key] = ([],[])
                if key in atomtype_embedding_perelem_dict:
                    atomtype_embedding_perelem_dict[key][0].extend(atypes)
                    atomtype_embedding_perelem_dict[key][1].extend(embsofatype)
    
    #print(atomtype_embedding_perelem_dict['c'][0])
    print(f"lens of the different lists: {len(atomtype_embedding_perelem_dict['c'][0])} {len(atomtype_embedding_perelem_dict['c'][1])}")
    #print(f"shapes of the different lists: {shape(atomtype_embedding_perelem_dict['c'][0])} {shape(atomtype_embedding_perelem_dict['c'][1])}")
    ############### P F Cl 
    namestring="c"
    plot_umap_pca_lda(atomtype_embedding_perelem_dict[namestring][1], atomtype_embedding_perelem_dict[namestring][0], namestring, save_path_prefix, colordict, min_dist, n_neighbors, alpha)
    """
    ############## P F Cl O -
    namestring="pfclo"
    plot_umap_pca_lda(p_f_cl_o_list_embs, p_f_cl_o_list_assigs, save_path_prefix, namestring, atomtype2color, min_dist, n_neighbors, alpha)
   
    ############## P F Cl S 
    namestring="pfcls"
    plot_umap_pca_lda(p_f_cl_s_list_embs, p_f_cl_s_list_assigs, save_path_prefix, namestring, atomtype2color, min_dist, n_neighbors, alpha)
    
    ############## C O
    namestring="co"
    plot_umap_pca_lda(c_o_list_embs, c_o_list_assigs, save_path_prefix, namestring, atomtype2color_co, min_dist, n_neighbors, alpha)

    print("Plotting................................BY ELEMENT")
    # plot all atomtypes of one element only
    for key in keylist:
        print(f"#######KEY {key}\n")
        pathway_umap = Path(str(save_path_prefix) +
                            f"umap_{min_dist}_{n_neighbors}_{key}_1.svg")
        pathway_pca = Path(str(save_path_prefix) + f"pca_{key}_1.svg")
        pathway_lda = Path(str(save_path_prefix) + f"lda_{key}_1")
        embeddings = dikt_forelems[key][0]
        assignments = dikt_forelems[key][1]
        atomtype2color, set_list = getcolorstoatomtype(set(assignments.copy()))

        try:
            assert len(embeddings) == (len(assignments)), "Assignments and embeddings do not have same length."
            assert len(embeddings)>10, "Not enough embeddings for plotting"
            print(f"len embeddings of key {key}: {len(embeddings)}")
            plot_pca(embeddings, assignments, atomtype2color, pathway_pca, alpha)
            plot_lda(embeddings, assignments, atomtype2color, pathway_lda, alpha)
            plot_umap(embeddings, assignments, atomtype2color, pathway_umap, min_dist, n_neighbors, alpha)
        except AssertionError as e:
            print(f"Assertion error occurred for element {key}: {e}")
            continue """


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
    dikt, totalfails, failedSmiPos, posToKeep_list = load_assignments_from_folder("./delaney_mols_bccc0_gaff2_assigned", smiles_dict, task_SMILES)
    
    #get embeddings from model
    finetuned_TASK_MODEL_PATH = Path("/data2/jgut/SoS_models")
    pretrained_TASK_MODEL_PATH = Path("/data/jgut/SMILES_or_SELFIES/prediction_models")
    model = "BART"
    traintype = "finetuned"
    embeds = get_embeddings_from_model(task, finetuned_TASK_MODEL_PATH, traintype, model, smiles_dict)

    #get rid of embeddings that encode for digits or hydrogens
    embeds_clean = get_clean_embeds(embeds, dikt, totalfails, task_SMILES)
    # within the dikt, map embeddings to atom types
    map_embeddings_to_atomtypes(dikt,task_SMILES)
    # following this, the dict looks as follows:     
    # dikt[SMILES] with dict_keys(['posToKeep', 'smi_clean', 'atom_types', 'max_penalty', 'orig_embedding', 'clean_embedding', 'atomtype_to_embedding'])
    
    unique_atomtype_set = set(chain.from_iterable(dikt[key]['atom_types'] for key in dikt if dikt[key].get('atom_types') is not None))
    atomtypes_to_elems_dict = create_elementsubsets(unique_atomtype_set)

    # get colors for atomtypes by element and element groups
    colordict = colorstoatomtypesbyelement(atomtypes_to_elems_dict)

    # plot embeddings
    min_dist = 0.1
    n_neighbors = 15
    alpha = 0.8
    penalty_threshold = 300
    save_path_prefix = f"./4July_{model}_{traintype}_thresh{penalty_threshold}/"
    create_plotsperelem(dikt, colordict, penalty_threshold, min_dist, n_neighbors, alpha, save_path_prefix)
    
    
    
    