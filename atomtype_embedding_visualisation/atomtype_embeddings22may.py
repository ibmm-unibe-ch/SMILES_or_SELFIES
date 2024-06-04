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
from preprocessing import canonize_smile
from tokenisation import tokenize_dataset, get_tokenizer
from constants import SEED
from sklearn.decomposition import PCA


from constants import (
    TASK_MODEL_PATH,
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

def smiles_to_file(smiles, no, ftype):
    """Execution of babel - generation  of file from SMILES

    Args:
        smiles (_string_): SMILES
        no (_int_): Number that goes in input file name
        ftype (_string_): pdb or mol2

    Returns:
        _string_: Resulting filename or None if execution fails
    """
    if ftype == "pdb":
        os.system(f'obabel -:"{smiles}" -o pdb -O ./mols/mol_{no}.pdb')
        return f"./mols/mol_{no}.pdb"
    elif ftype == "mol2":
        os.system(
            f'obabel -:"{smiles}" -o mol2 -O ./mols/mol_{no}.mol2 --gen3d')
        return f"./mols/mol_{no}.mol2"
    else:
        print("Execution of obabel failed, no file could be created. Wrong filetype given. Output filetype needs to be pdb or mol2.")
        return None


def exec_antechamber(inputfile, ftype):
    """Execution of antechamber - atomtype assignment of atoms from file

    Args:
        inputfile (_string_): Name of the inputfile
        ftype (_string_): Filtetype of the input: pdb or mol2

    Returns:
        _string_: Name of resulting antechamber-file or None if assignment failed.
    """
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
    """Checking of antechamber-assignment file with parmchk2

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
    """Running of parmchk2 to check assignment of files with antechamber

    Args:
        ac_outfile (_string_): Name of antechamber-file that contains assigned atoms

    Returns:
        _bool_: True if parmchk2 file is ok, False if it calls for revision of antechamber file
    """
    acout_noex = os.path.splitext(ac_outfile)[0]
    print("acout_noex", acout_noex)
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


def load_assignments_from_folder(folder, smiles_arr, smi_toks):
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
    assignment_list = list()
    dikt = dict()
    dikt_clean = dict()
    posToKeep_list = list()
    mol2_files = list()
    cleanSmis = list()
    assignment_fail = 0
    for file in os.listdir(folder):
        if file.endswith(".mol2") and not file.endswith("ass.mol2"):
            mol2_files.append(file)
    # sort according to numbers
    mol2_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    # print(mol2_files)
    #print("len of mol2files: ",len(mol2_files))

    filecreation_fail = len(smiles_arr)-(len(mol2_files))
    # I assume all file creations worked, if not this fails.
    assert(len(mol2_files) == (len(smiles_arr))
           ), f"Not every SMILES ({len(smiles_arr)}) has a corresponding file ({len(mol2_files)}) created for it. Needs more checking."
    for mol2 in mol2_files:
        num = int((re.findall(r'\d+', mol2.split('.')[0]))[0])
        print(num)
        parmcheck_file = f"mol_{num}_ass.frcmod"
        assignment_file = f"mol_{num}_ass.mol2"
        smi = smiles_arr[num]
        smi_clean, posToKeep = clean_SMILES(smi_toks[num])

        print(f"num {num} extracted from mol2 {mol2}")
        # check whether assignment exists
        if os.path.isfile(f"{folder}/{assignment_file}") == True:
            print("assignment exists")
            # check whether parmcheck is ok, if yes, save output of assignment file
            if os.path.isfile(f"{folder}/{parmcheck_file}") == True:
                print("parmchk file exists and is ok")
                if check_parmchk2(f"{folder}/{parmcheck_file}") == True:
                    # get atom assignments from ass file
                    atoms_assignment_list, atoms_assignment_set = get_atom_assignment(
                        f"{folder}/{assignment_file}")
                    assignment_list.append(atoms_assignment_list)
                    dikt[smi] = (posToKeep, smi_clean, atoms_assignment_list)
                    dikt_clean[smi] = (posToKeep, smi_clean,
                                       atoms_assignment_list)
                    posToKeep_list.append(posToKeep)
                else:
                    dikt[smi] = (None, None, None)
                    failedSmiPos.append(num)
                    assignment_fail += 1
            else:
                dikt[smi] = (None, None, None)
                failedSmiPos.append(num)
                assignment_fail += 1
        else:
            dikt[smi] = (None, None, None)
            failedSmiPos.append(num)
            assignment_fail += 1

    cleanSmis = list()
    for smi, smi_tok in zip(smiles_arr, smi_toks):
        print("##############################################################################################################")
        print(f"SMILES: {smi}")
        smi_clean, posToKeep = clean_SMILES(smi_tok)
        cleanSmis.append(smi_clean)

    assert(len(dikt.keys()) == (len(smiles_arr))
           ), f"Number of keys for SMILES in dictionary ({len(dikt.keys())}) not equal to number of SMILES in original array ({len(smiles_arr)})"
    assert(len(dikt.keys()) == (len(smi_toks))
           ), f"Number of keys for SMILES in dictionary ({len(dikt.keys())}) not equal to number of SMILES in original array of tokens ({len(smi_toks)})"
    assert len(posToKeep_list) == len(dikt_clean.keys(
    )), f"Length of list of positions of assigned atoms in SMILES ({len(posToKeep_list)}) and number of SMILES ({len(posToKeep_list)}) is not the same."
    logging.info(
        f"File creation from SMILES to pdb by obabel failed {filecreation_fail} times out of {len(rndm_smiles)}")
    logging.info(
        f"Atom assignment by antechamber failed {assignment_fail} times out of {len(rndm_smiles)}")

    return dikt, dikt_clean, posToKeep_list, filecreation_fail+assignment_fail, failedSmiPos, cleanSmis


def get_atom_assignments(smiles_arr, smi_toks):
    """Getting the atom assignments

    Args:
        smiles_arr (_list_): Array of SMILES
        smi_toks (_list_): List of lists that corresponds to smiles_arr and contains the tokens to the corresponding SMILES

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
        smi_clean, posToKeep = clean_SMILES(smi_tok)
        cleanSmis.append(smi_clean)
        #print(f"smi_tok turns to smi_clean: {smi_tok}  --->  {smi_clean}")
        smi_fi = smiles_to_file(smi, no, "mol2")
        if os.path.isfile(smi_fi) == True:
            print("Successful conversion of SMILES to file")
            smi_ac = exec_antechamber(smi_fi, "mol2")
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
    print("model loaded")
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


def get_clean_embeds(embeds, failedSmiPos, posToKeep_list, creation_assignment_fail):
    """Clean embeddings of embeddings that encode for digitis or hydrogens

    Args:
        embeds (_List[List[float]_): Embeddings of a SMILES
        failedSmiPos (_list_): Positions of SMILES in list where no file and/or assignment could be generated
        posToKeep_list (_list_): List of positions in a SMILES according to tokens that need to be kept (not digits or hydrogens)

    Returns:
        _list[float]_: Embeddings that do not encode hydrogens or digits, but only atoms
    """

    embeds_clean = list()
    for count, emb in enumerate(embeds[0]):
        if count not in failedSmiPos:  # assignment for this SMILES did not fail
            embeds_clean.append(emb)
    logging.info(
        f"Length embeddings before removal: {len(embeds[0])}, after removal where atom assignment failed: {len(embeds_clean)}")
    assert creation_assignment_fail == (len(
        embeds[0])-len(embeds_clean)), f"Assignment fails ({creation_assignment_fail}) and number of deleted embeddings do not agree ({(len(embeds[0])-len(embeds_clean))})."

    embeds_cleaner = []
    assert len(embeds_clean) == (len(posToKeep_list)
                                 ), f"Not the same amount of embeddings as assigned SMILES. {len(embeds_clean)} embeddings vs. {(len(posToKeep_list))} SMILES with positions"
    #assuring that only embeddings of atoms are kept according to posToKeep_list
    for smiemb, pos_list in zip(embeds_clean, posToKeep_list):
        newembsforsmi = []
        newembsforsmi = [smiemb[pos] for pos in pos_list]
        embeds_cleaner.append(newembsforsmi)

    # sanity check that the lengths agree
    for smiemb, pos_list in zip(embeds_cleaner, posToKeep_list):
        assert len(smiemb) == len(
            pos_list), "Final selected embeddings for assigned atoms do not have same length as list of assigned atoms."
    return embeds_cleaner


def check_lengths(smi_toks, embeds):
    """Check that number of tokens corresponds to number of embeddings, otherwise sth went wrong

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


def link_embeds_to_atomassigns(embeds_clean, smiToAtomAssign_dict_clean):
    """Linking the embeddings to their atom assignments

    Args:
        embeds_clean (list[float]): Embeddings that do not encode hydrogens or digits, but only atoms
        smiToAtomAssign_dict_clean (dict): Dictionary that links SMILES to their atom assignments

    Returns:
        _dict_: Dictionary that links SMILES to their corresponding embeddings and assignments
    """
    embass_dikt = dict()
    assert (len(smiToAtomAssign_dict_clean.keys()) == (len(embeds_clean))
            ), f"Number of assigned SMILES ({len(smiToAtomAssign_dict_clean.keys())}) and embeddings {len(embeds_clean)} do not agree."
    it = 0
    for smi, value in smiToAtomAssign_dict_clean.items():
        clean_toks = value[1]
        assigns = value[2]
        assert len(clean_toks) == (len(
            embeds_clean[it])), f"Number of tokens ({len(clean_toks)}) does not equal number of embeddings ({len(embeds_clean[it])}) for this SMILES string"
        assert len(assigns) == (len(
            embeds_clean[it])), f"Number of assignments ({len(assigns)}) does not equal number of embeddings ({len(embeds_clean[it])}) for this SMILES string.\n Assigns: {assigns} vs. Embeddings"
        embass_dikt[smi] = (embeds_clean[it], assigns)
        it += 1
    return embass_dikt


def build_legend(data):
    """
    Build a legend for matplotlib plt from dict
    """
    legend_elements = []
    for key in data:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=key,
                                      markerfacecolor=data[key], markersize=9))
    return legend_elements


def plot_umap(embeddings, labels, colours_dict, set_list, save_path, min_dist=0.1, n_neighbors=15, alpha=0.2):
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
    ax.set_ylabel("UMAP 2")
    ax.set_xlabel("UMAP 1")
    ax.set_title("UMAP - Embeddings resp. to atom types")
    fig.savefig(save_path, format="svg")
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
    fig, ax = plt.subplots(1)
    ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], marker='.', alpha=alpha, c=[
               colours_dict[x] for x in labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(1.13, 0.5), fontsize=8)
    ax.set_ylabel("PCA 2")
    ax.set_xlabel("PCA 1")
    ax.set_title("PCA - Embeddings resp. atom types")
    fig.savefig(save_path, format="svg")
    fig.clf()


def getcolorstoatomtype(big_set):
    """Generating a dictionary of colors given a set of atom types

    Args:
        big_set (_set_): Set of atom types

    Returns:
        _dict,set_: Dictionary that maps atom types to colors, alphabetically sorted list of input set
    """
    #cmap = mpl.cm.get_cmap('viridis')
    #nums = np.linspace(0,1.0,(len(big_set)))
    #colors_vir = [cmap(num) for num in nums]
    # https://sashamaps.net/docs/resources/20-colors/ #95% accessible only, subject to change
    colors_sash = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
                   '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']

    atomtype2color = dict()
    # sort big set according to alphabet to keep carbons closer together
    set_list = sorted(list(big_set))

    for atype, col in zip(set_list, colors_sash[0:len(set_list)]):
        atomtype2color[atype] = col

    return atomtype2color, set_list


def create_elementsubsets(big_set, embeds_fin_singlelist, atom_assigns_fin_singlelist):
    """Creation of element subsets according to alphabet

    Args:
        big_set (_set_): Set of atom types
        embeds_fin_singlelist (_list[float]_): List of embeddings
        atom_assigns_fin_singlelist (_list[string]_): List of atom assignments

    Returns:
        _list,dict[string][list[float],list[string]]_: List of keys (elements), dictionary that contains embeddings and their atomtypes sorted by element
    """
    dikt = dict()
    last_firstval = ''
    curr_liste = list()
    ctr = 0
    #print(big_set)
    for s in big_set:
        if ctr == 0:
            last_firstval = s[0]
        # cl will be treated differently
        if last_firstval != s[0] or ctr == len(big_set)-1 and s != 'cl':
            dikt[last_firstval] = curr_liste
            curr_liste = list()
            last_firstval = s[0]
        if last_firstval == s[0]:
            if s != 'cl':  # cl will be treated differently
                curr_liste.append(s)
        ctr += 1
    dikt['cl'] = ['cl']  # cl
    print(dikt.items())

    it = 0
    keylist = dikt.keys()
    print("keylist", keylist)
    dikt_forelems = dict()
    for elem in keylist:
        curr_emblist = list()
        curr_assignlist = list()
        for assign, emb in zip(atom_assigns_fin_singlelist, embeds_fin_singlelist):
            if elem == assign[0] and assign != 'cl':
                curr_emblist.append(emb)
                curr_assignlist.append(assign)
        dikt_forelems[elem] = (curr_emblist, curr_assignlist)
    curr_emblist = list()
    curr_assiglist = list()
    for assig, emb in zip(atom_assigns_fin_singlelist, embeds_fin_singlelist):
        if assig == 'cl':
            curr_emblist.append(emb)
            curr_assiglist.append(assig)
    dikt_forelems['cl'] = (curr_emblist, curr_assiglist)
    print("dikt for elems keys:", dikt_forelems.keys())
    assert len(keylist) == (len(dikt_forelems.keys())
                            ), "Keylist and list of elements in dict not the same."
    return keylist, dikt_forelems


def create_plotsperelem(keylist, dikt_forelems, min_dist, n_neighbors, alpha, save_path_prefix):
    """Create plot per element

    Args:
        keylist (_list_): List of keys/elements
        dikt_forelems (_dict[string][list[float],list[string]]): Dictionary that contains ambeddings and their atomtypes sorted by element
        min_dist (_float_): Number of min dist to use in UMAP
        n_neighbors (_int_): Number of neighbors to use in UMAP
        alpha (_int_): Level of opacity
        save_path_prefix (_string_): Path prefix where to save output plot
    """
    print("Plotting................................P F Cl plots")
    #plots for all elements p, f, and cl
    #get seperate embedding lists for plotting
    p_f_list_embs = (dikt_forelems['p'][0]) + dikt_forelems['f'][0]
    p_f_cl_list_embs = p_f_list_embs + (dikt_forelems['cl'][0])
    p_f_cl_o_list_embs = p_f_cl_list_embs + (dikt_forelems['o'][0])
    p_f_cl_s_list_embs = p_f_cl_list_embs + (dikt_forelems['s'][0])
    
    #get seperate assignment lists for plotting
    p_f_list_assigs = (dikt_forelems['p'][1]) + dikt_forelems['f'][1]
    p_f_cl_list_assigs = p_f_list_assigs + (dikt_forelems['cl'][1])
    p_f_cl_o_list_assigs = p_f_cl_list_assigs + (dikt_forelems['o'][1])
    p_f_cl_s_list_assigs = p_f_cl_list_assigs + (dikt_forelems['s'][1])
    
    #sanity check
    assert len(p_f_cl_list_embs) == len(p_f_cl_list_assigs)
    assert len(p_f_cl_o_list_embs) == len(p_f_cl_o_list_assigs)
    assert len(p_f_cl_s_list_embs) == len(p_f_cl_s_list_assigs)
    print("assiglist", p_f_cl_list_assigs)
    ############### P F Cl -->  get distinct colors for all atomtypes for elements p, f, cl
    atomtype2color, set_list = getcolorstoatomtype(set(p_f_cl_list_assigs.copy()))
    #create paths on what to name the plots
    pathway = Path(str(save_path_prefix) +
                   f"umap_{min_dist}_{n_neighbors}_pfcl_1.svg")
    pathway_pca = Path(str(save_path_prefix) + "pca_pfcl_1.svg")
    # plot UMAP
    plot_umap(p_f_cl_list_embs, p_f_cl_list_assigs, atomtype2color,
              set_list, pathway, min_dist, n_neighbors, alpha)
    # plot PCA
    plot_pca(p_f_cl_list_embs, p_f_cl_list_assigs,
             atomtype2color, pathway_pca, alpha)
    ############## P F Cl O --> get distinct colors for all atomtypes for elements p, f, cl, o
        #create paths on what to name the plots
    pathway = Path(str(save_path_prefix) +
                   f"umap_{min_dist}_{n_neighbors}_pfclo_1.svg")
    pathway_pca = Path(str(save_path_prefix) + "pca_pfclo_1.svg")
    atomtype2color, set_list = getcolorstoatomtype(set(p_f_cl_o_list_assigs.copy()))
    # plot UMAP
    plot_umap(p_f_cl_o_list_embs, p_f_cl_o_list_assigs, atomtype2color,
              set_list, pathway, min_dist, n_neighbors, alpha)
    # plot PCA
    plot_pca(p_f_cl_o_list_embs, p_f_cl_o_list_assigs,
             atomtype2color, pathway_pca, alpha)
    ############## P F Cl S --> get distinct colors for all atomtypes for elements p, f, cl, o
        #create paths on what to name the plots
    pathway = Path(str(save_path_prefix) +
                   f"umap_{min_dist}_{n_neighbors}_pfcls_1.svg")
    pathway_pca = Path(str(save_path_prefix) + "pca_pfcls_1.svg")
    atomtype2color, set_list = getcolorstoatomtype(set(p_f_cl_s_list_assigs.copy()))
    # plot UMAP
    plot_umap(p_f_cl_s_list_embs, p_f_cl_s_list_assigs, atomtype2color,
              set_list, pathway, min_dist, n_neighbors, alpha)
    # plot PCA
    plot_pca(p_f_cl_s_list_embs, p_f_cl_s_list_assigs,
             atomtype2color, pathway_pca, alpha)
    

    print("Plotting................................BY ELEMENT")
    # plot all atomtypes of one element only
    for key in keylist:
        print(f"#######KEY {key}\n")
        pathway_umap = Path(str(save_path_prefix) +
                            f"umap_{min_dist}_{n_neighbors}_{key}_1.svg")
        pathway_pca = Path(str(save_path_prefix) + f"pca_{key}_1.svg")
        embeddings = dikt_forelems[key][0]
        assignments = dikt_forelems[key][1]
        atomtype2color, set_list = getcolorstoatomtype(set(assignments.copy()))

        try:
            assert len(embeddings) == (len(assignments)), "Assignments and embeddings do not have same length."
            assert len(embeddings)>10, "Not enough embeddings for plotting"
            print(f"len embeddings of key {key}: {len(embeddings)}")
            plot_pca(embeddings, assignments, atomtype2color, pathway_pca, alpha)
            plot_umap(embeddings, assignments, atomtype2color, set_list,
                    pathway_umap, min_dist, n_neighbors, alpha)
        except AssertionError as e:
            print(f"Assertion error occurred for element {key}: {e}")
            continue



if __name__ == "__main__":

    ############################### get SMILES from Task ###################################################   
    task = "delaney"
    #task = "bace_classification"
    """     assert task in list(
        MOLNET_DIRECTORY.keys()
    ), f"{task} not in MOLNET tasks."
    """
    task_SMILES, task_labels = load_molnet_test_set(task)
    print(
        f"SMILES: {task_SMILES} \n len task_SMILES delaney: {len(task_SMILES)}")
    #print(f"task labels",task_labels)
    rndm_smiles = task_SMILES
    print(f"first smiles {task_SMILES[0]} and length {len(task_SMILES[0])}")
    print("delaney reading done")
    
    ###############################get tokenized version of dataset ########################################
    # get tokenized version of dataset
    tokenizer = get_tokenizer(TOKENIZER_PATH)
    print(f"tokenizer {tokenizer}")
    smi_toks = tokenize_dataset(tokenizer, task_SMILES, False)
    print("whole SMILES tokenized: ",smi_toks[0])
    smi_toks = [smi_tok.split() for smi_tok in smi_toks]
    print(f"SMILES tokens after splitting tokens into single strings: {smi_toks[0]}")


    ############################## get atomassignments for Delaney using antechamber and parmchk2 OR from previous antechamber assignment with antechamber and parmchk2 ########################################
        # get atom assignments from SMILES
    #smiToAtomAssign_dict, smiToAtomAssign_dict_clean, posToKeep_list, creation_assignment_fail, failedSmiPos, cleanSmis = get_atom_assignments(task_SMILES,smi_toks)   
    #
    # get atom assignments from folder, returns what positions are kept from original SMILES, 
    smiToAtomAssign_dict, smiToAtomAssign_dict_clean, posToKeep_list, creation_assignment_fail, failedSmiPos, cleanSmis = load_assignments_from_folder(
        "./delaney_mols_bccc0_gaff2_assigned", task_SMILES, smi_toks)
    ######exemplary smiToAtomAssign_dict_clean: ('c1cc2ccc3cccc4ccc(c1)c2c34', ###############
        # ([0, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 21, 23], --> positions to be kept from original SMILES 
        # ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c'], --> cleaned SMILES
        # ['ca', 'ca', 'ca', 'ca', 'ca', 'ca', 'ca', 'ca', 'ca', 'ca', 'ca', 'ca', 'ca', 'ca', 'ca', 'ca'])) --> atomtypes for each element according to Antechamber 
    #############example end####################################################
    #print the clean SMILES to Atom Assignment dictionary
    #print()
    #for key, value in smiToAtomAssign_dict_clean.items():
    #    print(f"Key: {key}")
    #    print(f"Value: {value}")
    #print()
    print("#############first key and value in smiToAtomAssign_dict_clean: ",list(smiToAtomAssign_dict_clean.items())[0])
    
    ############################## get embeddings per token ########################################
    # get embeddings per token
    # paths for BART   
    """   specific_model_path = (
        TASK_MODEL_PATH
        / task
        / "smiles_atom_isomers_bart"
        / "1e-05_0.2_seed_0" 
        / "checkpoint_best.pt"
    )
    data_path = TASK_PATH / task / "smiles_atom_isomers" 
    """
    #paths for RoBERTa
    specific_model_path = (
        TASK_MODEL_PATH
        / task
        / "smiles_atom_isomers_roberta"
        / "1e-05_0.2_seed_0" 
        / "checkpoint_best.pt"
    )
    print("specific model path: ",specific_model_path)
    data_path = TASK_PATH / task / "smiles_atom_isomers"
    
    embeds = []
    embeds = get_embeddings(task, specific_model_path, data_path, False) #works for BART model with newest version of fairseq on github, see fairseq_git.yaml file
    print("got the embeddings")
    
    # check their lengths, dict contains SMILES where atoms could not be assigned as well, embeddings we got for all SMILES
    assert len(smiToAtomAssign_dict.keys()) == (len(
        embeds[0])), f"Number of SMILES and embeddings do not agree. Number of SMILES: {len(smiToAtomAssign_dict.keys())} and Number of embeddings: {len(embeds[0])}"
    #check that every token has an embedding
    assert check_lengths(
        smi_toks, embeds), "Length of SMILES_tokens and embeddings do not agree."
    print("embeddings passed length checks")
    
    #get rid of embeddings that encode for digits or hydrogens
    embeds_clean = get_clean_embeds(embeds, failedSmiPos, posToKeep_list,creation_assignment_fail)

    ############################## creating dictionary mapping SMILES to embeddings and assignments ########################################
    # map embeddings to atom assignments --> Dictionary that links SMILES to their corresponding embeddings and assignments
    # --> smiles = ([[embed], smilesatomname], [atomassignments])
    embass_dikt = link_embeds_to_atomassigns(
        embeds_clean, smiToAtomAssign_dict_clean)
    first_key, first_value = list(embass_dikt.items())[0]
    print(f" embass_dikt: first key: {first_key}, len key: {len(first_key)},\n len first value: {len(first_value[0])},  \n len first value[0]: {len(first_value[0][0])},\n first value: {first_value[0][0]},\n len second value: {len(first_value[1])}")
    
    #getting all embeddings, all assignmnets, and numbers for all
    embeds_fin = [val[0] for val in embass_dikt.values()]
    atom_assigns_fin = [val[1] for val in embass_dikt.values()]
    mol_labels = [num for num in range(0, len(embeds_fin))]
 
    # sanity check
    for emb, assign in zip(embeds_fin, atom_assigns_fin):
        assert len(emb) == (len(
            assign)), f"embeddings for smi and assignments do not have same length: {len(emb)} vs {len(assign)}"
    print("embeddings and assignments have same length")
    
    #################################### Creating big final lists ############################################################################
    ############# creating a single big final list of embeddings that only contains embeddings (no atom names)
    embeds_fin_singlelist = list()
    for smiembed in embeds_fin:
        for atomembed in smiembed:
            embeds_fin_singlelist.append(atomembed[0])
    print(f"final list of embeddings/points: {len(embeds_fin_singlelist)}")

    ############# creating a single big final list of all atom assignments
    atom_assigns_fin_singlelist = list()
    for smiassigns in atom_assigns_fin:
        for singleatomassign in smiassigns:
            atom_assigns_fin_singlelist.append(singleatomassign)
    print(f"final list of assignments: {len(atom_assigns_fin_singlelist)}")

    ################################### Create sets to cluster atom types and assign colors to them ############################################
    # create a set from atom types for each list, create a greater set from it, assign each atom type a color
    atomtype_set = [set(type_list) for type_list in atom_assigns_fin]
    big_set = set().union(*atomtype_set)
    big_set = sorted(list(big_set))
    print("set of atomtypes in dataset sorted alphabetically: ", big_set)

    
    # for atoms in big set create separate embedding lists
    # sorting according to first letter of atom type, but Cl does not belong to carbon
    # dikt_for_elements clusters embeddings corresponding atom assignments according to elements and not atomtypes e.g. [('c', ['c', 'c1', 'c2', 'c3', 'ca', 'cc', ...]), ('cl'), ['cl'])], ...)
    # keylist just contains the keys: dict_keys(['c', 'f', 'n', 'o', 'p', 's', 'cl'])
    keylist, dikt_forelems = create_elementsubsets(
        big_set, embeds_fin_singlelist, atom_assigns_fin_singlelist)
    
    
    ################################### Plotting the embeddings per element and saving them as .svg ############################################
    min_dist = 0.1
    n_neighbors = 15
    alpha = 0.8
    save_path_prefix = f"./plots_RoBERTa/{task}"
    create_plotsperelem(keylist, dikt_forelems, min_dist,
                        n_neighbors, alpha, save_path_prefix)
                        
