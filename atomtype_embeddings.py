import pandas as pd
import numpy as np
import os
from typing import List, Tuple
from io import StringIO
import logging
import umap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path
import re
from string import ascii_lowercase as alc

from fairseq_utils import compute_model_output, load_dataset
from fairseq.data import Dictionary
from scoring import load_model
from attention_readout import load_molnet_test_set, canonize_smile
from plotting import plot_representations
from tokenisation import tokenize_dataset, get_tokenizer
from constants import SEED
from sklearn.decomposition import PCA

from constants import (
    TASK_MODEL_PATH,
    TASK_PATH,
    MOLNET_DIRECTORY,
    TOKENIZER_PATH
)
markers = list(Line2D.markers.keys())
prop_cycle = plt.rcParams["axes.prop_cycle"]
default_colours = prop_cycle.by_key()["color"]


def smilestofile(smiles,no,ftype):
    """Execution of babel - generatio  of file from SMILES

    Args:
        smiles (_string_): _description_
        no (_int_): Number that goes in input file name
        ftype (_string_): pdb or mol2

    Returns:
        _string_: Resulting filename or None if execution fails
    """
    #obabel [-i<input-type>] <infilename> [-o<output-type>] -O<outfilename> [Options]
    if ftype=="pdb":
        os.system(f'obabel -:"{smiles}" -o pdb -O ./mols/mol_{no}.pdb')
        return f"./mols/mol_{no}.pdb"
    elif ftype=="mol2":
        os.system(f'obabel -:"{smiles}" -o mol2 -O ./mols/mol_{no}.mol2 --gen3d')
        return f"./mols/mol_{no}.mol2"
    else:
        print("Execution of obabel failed, no file could be created. Wrong filetype given. Output filetype needs to be pdb or mol2.")
        return None

#call 
def exec_antechamber(inputfile,ftype):
    """Execution of antechamber - atomtype assignment of atoms from file

    Args:
        inputfile (_string_): Name of the inputfile
        ftype (_string_): Filtetype of the input: pdb or mol2

    Returns:
        _string_: Name of resulting antechamber-file or None if assignment failed.
    """
    inputfile_noex=os.path.splitext(inputfile)[0]
    #print("inputfile no extension",inputfile_noex)
    if ftype=="pdb":
        #os.system(f"antechamber -i {inputfile} -fi pdb -o {inputfile_noex}_ass.mol2 -fo mol2 -c bcc -nc 0 -at gaff2")
        os.system(f"antechamber -i {inputfile} -fi pdb -o {inputfile_noex}_ass.mol2 -fo mol2 -c bcc -nc 0 -at gaff2")
    elif ftype=="mol2":
        os.system(f"antechamber -i {inputfile} -fi mol2 -o {inputfile_noex}_ass.mol2 -fo mol2 -c bcc -nc 0 -at gaff2")
    else:
        print("Execution of antechamber failed. Wrong filetype given. Filetype needs to be pdb or mol2.")
        return None
    return f"{inputfile_noex}_ass.mol2"

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
                    logging.info(f"Atom assignment failed: parmchk2 calls for revision for file {file}")
                    print("###############################################################################")
                    print("###############################################################################")
                    print("################################ATTENTION######################################")
                    print("#####################Parmchk2: atomtypes need revision#########################")
                    print("###############################################################################")
                    return False
            return True

def run_parmchk2(ac_outfile):
    """Running of parmchk2 to check assignment of files with antechamber

    Args:
        ac_outfile (_string_): Name of antechamber-file that contains assigned atoms

    Returns:
        _bool_: True if parmchk2 file is ok, False if it calls for revision of antechamber file
    """
    acout_noex=os.path.splitext(ac_outfile)[0]
    print("acout_noex",acout_noex)
    os.system(f"parmchk2 -i {ac_outfile} -f mol2 -o {acout_noex}.frcmod -s gaff2")
    #check whether file was generated
    if os.path.isfile(f"{acout_noex}.frcmod")==True:
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
    ac_out_noH=list()
    for j in ac_out:
        if not j.startswith('H') and not j.startswith('h'):
            ac_out_noH.append(j)
    print("before: ", ac_out)
    print("after: ", ac_out_noH)
    return ac_out_noH

def getatom_ass(mol2):
    """Extracting the assignment of atom types to atoms from antechamber output

    Args:
        mol2 (_string_): Name of antechamber-mol2-outputfile

    Returns:
        _list,set_: List of assigned atom types without cleared of hydrogens and set of assigned atomtypes
    """
    #extract lines between @<TRIPOS>ATOM and @<TRIPOS>BOND to get atom asss
    with open(mol2) as infile:
        lines = infile.read().splitlines()
    start = [i for i, line in enumerate(lines) if line.startswith("@<TRIPOS>ATOM")][0]
    end = [i for i, line in enumerate(lines) if line.startswith("@<TRIPOS>BOND")][0]
    extract="\n".join(lines[start+1:end])
    print("\n_______________________extraction \n", extract)
    pddf = pd.read_csv(StringIO(extract), header=None, delimiter=r"\s+")
    #extract 5th column with atom_asss
    atoms_ass_list = pddf.iloc[:,5].tolist()
    #clean H from atom assignment
    atoms_ass_list_clean = clean_acout(atoms_ass_list)
    atoms_ass_set = set(atoms_ass_list_clean)
    return atoms_ass_list_clean, atoms_ass_set


def clean_SMILES(SMILES_tok):
    """Cleaning of SMILES tokens input from hydrogens and digits

    Args:
        SMILES_tok (_list_): List of SMILES_tokens for a given SMILES

    Returns:
        _list,list_: Processed SMILES_token list and list of positions in input tokens list that were kept 
        (needed to distinguish which embeddings are relevant)
    """
    SMILES_tok_prep=list()
    struc_toks=r"()=:~1234567890#"
    posToKeep=list()
    pos=0
    for i in range(len(SMILES_tok)):
        #when it's an H in the SMILES, ignore, cannot deal
        #print(SMILES_tok[i])
        if SMILES_tok[i]!="H" and SMILES_tok[i]!="h" and not SMILES_tok[i].isdigit() and not SMILES_tok[i].isspace():
            if any(elem in struc_toks for elem in SMILES_tok[i])==False:
                if SMILES_tok[i]!="-":
                    SMILES_tok_prep.append(SMILES_tok[i])
                    posToKeep.append(pos) #keep pos where you keep SMILES token
        pos+=1
    assert(len(posToKeep)==(len(SMILES_tok_prep)))
    print("SMILES_tok: ", SMILES_tok )
    print("posToKeep: ",posToKeep)
    print("SMILES_tok_prep: ",SMILES_tok_prep)
    
    return SMILES_tok_prep,posToKeep

def load_assignments_from_folder(folder,smiles_arr,smi_toks):
    """Function to load atom assignments from folder given foder with mol2-files, atomassignment outputs from antechamber 
    and parmchk2-outputfiles that were created when checking the antechamber output

    Args:
        folder (_string_): Name of folder from which to load files
        smiles_arr (_list_): Array of SMILES
        smi_toks (_list_): List of lists that corresponds to smiles_arr and contains the tokens to th corresponding SMILES

    Returns:
        _dict,dict,list,int,list,list_: Many: dikt, dikt_clean, posToKeep_list, filecreation_fail+assignment_fail, failedSmiPos, cleanSmis
    """
    failedSmiPos=list()
    assignment_list=list()
    dikt=dict()
    dikt_clean = dict()
    posToKeep_list = list()
    mol2_files=list()
    cleanSmis=list()
    assignment_fail=0
    for file in os.listdir(folder):
        if file.endswith(".mol2") and not file.endswith("ass.mol2"):
            mol2_files.append(file)
    mol2_files.sort(key=lambda f: int(re.sub('\D', '', f))) #sort according to numbers
    print(mol2_files)
    print("len of mol2files: ",len(mol2_files))

    filecreation_fail = len(smiles_arr)-(len(mol2_files))
    assert(len(mol2_files)==(len(smiles_arr))), "Not every SMILES has a corresponding file created for it. Needs more checking." #I assume all file creations worked, if not this fails.
    for mol2 in mol2_files:
        num = int((re.findall(r'\d+', mol2.split('.')[0]))[0])
        print(num)
        parmcheck_file = f"mol_{num}_ass.frcmod" 
        assignment_file = f"mol_{num}_ass.mol2"
        smi = smiles_arr[num]
        smi_clean, posToKeep = clean_SMILES(smi_toks[num])
        
        print(f"num {num} extracted from mol2 {mol2}")
        #check whether assignment exists
        if os.path.isfile(f"{folder}/{assignment_file}")==True:
            print("assignment exists")
            #check whether parmcheck is ok, if yes, save output of assignment file
            if os.path.isfile(f"{folder}/{parmcheck_file}")==True:
                print("parmchk file exists and is ok")
                if check_parmchk2(f"{folder}/{parmcheck_file}")==True:
                    #get atom assignments from ass file
                    atoms_ass_list, atoms_ass_set = getatom_ass(f"{folder}/{assignment_file}")
                    assignment_list.append(atoms_ass_list)
                    dikt[smi] = (posToKeep,smi_clean,atoms_ass_list)
                    dikt_clean[smi] = (posToKeep,smi_clean,atoms_ass_list)
                    posToKeep_list.append(posToKeep)
                else:
                    dikt[smi] = (None,None,None)
                    failedSmiPos.append(num)
                    assignment_fail+=1  
            else:
                dikt[smi] = (None,None,None)
                failedSmiPos.append(num)
                assignment_fail+=1  
        else:
            dikt[smi] = (None,None,None)
            failedSmiPos.append(num)
            assignment_fail+=1     
            
    cleanSmis=list()
    for smi,smi_tok in zip(smiles_arr,smi_toks):
        print("##############################################################################################################")
        print(f"SMILES: {smi}")
        smi_clean, posToKeep = clean_SMILES(smi_tok)
        cleanSmis.append(smi_clean)
        
    assert(len(dikt.keys())==(len(smiles_arr)))
    assert(len(dikt.keys())==(len(smi_toks)))
    assert len(posToKeep_list)==len(dikt_clean.keys()), f"Length of list of positions of assigned atoms in SMILES ({len(posToKeep_list)}) and number of SMILES ({len(posToKeep_list)}) is not the same."
    logging.info(f"File creation from SMILES to pdb by obabel failed {filecreation_fail} times out of {len(rndm_smiles)}")
    logging.info(f"Atom assignment by antechamber failed {assignment_fail} times out of {len(rndm_smiles)}")
    
    return dikt, dikt_clean, posToKeep_list, filecreation_fail+assignment_fail, failedSmiPos, cleanSmis

def get_atom_assignments(smiles_arr,smi_toks):
    """Getting the atom assignments

    Args:
        smiles_arr (_list_): Array of SMILES
        smi_toks (_list_): List of lists that corresponds to smiles_arr and contains the tokens to th corresponding SMILES

    Returns:
         _dict,dict,list,int,list,list_: Many: dikt, dikt_clean, posToKeep_list, filecreation_fail+assignment_fail, failedSmiPos, cleanSmis
    """
    no=0
    assignment_list=list()
    dikt=dict()
    dikt_clean = dict()
    posToKeep_list = list()
    filecreation_fail = 0
    assignment_fail = 0
    smi_num = 0
    failedSmiPos = list()
    cleanSmis = list()
    for smi,smi_tok in zip(smiles_arr,smi_toks):
        print("##############################################################################################################")
        print(f"SMILES: {smi}")
        smi_clean, posToKeep = clean_SMILES(smi_tok)
        cleanSmis.append(smi_clean)
        #print(f"smi_tok turns to smi_clean: {smi_tok}  --->  {smi_clean}")
        smi_fi = smilestofile(smi,no,"mol2")
        if os.path.isfile(smi_fi)==True:
            print("Successful conversion of SMILES to file")
            smi_ac = exec_antechamber(smi_fi,"mol2")
            if os.path.isfile(smi_ac)==True:
                #if smi_ac was generated check if with parmchk2, returns True if output is ok
                if True==run_parmchk2(smi_ac):
                    #get antechamber assignment (without hydrogens)
                    atoms_ass_list, atoms_ass_set = getatom_ass(smi_ac)    
                    assignment_list.append(atoms_ass_list)
                    dikt[smi] = (posToKeep,smi_clean,atoms_ass_list)
                    dikt_clean[smi] = (posToKeep,smi_clean,atoms_ass_list)
                    posToKeep_list.append(posToKeep)
                else:
                    assignment_fail +=1
                    dikt[smi] = (None,None,None)
                    failedSmiPos.append(smi_num)
            else:
                assignment_fail +=1
                dikt[smi] = (None,None,None)
                failedSmiPos.append(smi_num)
        else:
            filecreation_fail +=1
            dikt[smi] = (None,None,None) 
            failedSmiPos.append(smi_num)
        no+=1
        smi_num+=1
    assert(len(dikt.keys())==(len(smiles_arr)))
    assert(len(dikt.keys())==(len(smi_toks)))
    assert len(posToKeep_list)==len(dikt_clean.keys()), f"Length of list of positions of assigned atoms in SMILES ({len(posToKeep_list)}) and number of SMILES ({len(posToKeep_list)}) is not the same."
    logging.info(f"File creation from SMILES to pdb by obabel failed {filecreation_fail} times out of {len(rndm_smiles)}")
    logging.info(f"Atom assignment by antechamber failed {assignment_fail} times out of {len(rndm_smiles)}")
    return dikt, dikt_clean, posToKeep_list, filecreation_fail+assignment_fail, failedSmiPos, cleanSmis

def get_embeddings(task: str, cuda: int
) -> Tuple[List[List[float]], np.ndarray]:
    """Generate the attention dict of a task
    Args:
        task (str): Task to find attention of
        cuda (int): CUDA device to use
    Returns:
        Tuple[List[List[float]], np.ndarray]: attention, labels
    """
    task_SMILES, task_labels = load_molnet_test_set(task)
    for encoding in [
        "smiles_atom"
    ]:
        specific_model_path = (
            TASK_MODEL_PATH
            / task
      #      / "smiles_isomers_atom"
            / encoding
            / "5e-05_0.2_based_norm"
            / "5e-05_0.2_based_norm"
            / "checkpoint_best.pt"
        )
        data_path = TASK_PATH / task / encoding
        model = load_model(specific_model_path, data_path, cuda)
        model.zero_grad()
        data_path = data_path / "input0" / "test"
        dataset = load_dataset(data_path, True) #True for classification, false for regression
        #print(dataset)
        #print(len(dataset))
        #print("datapath for source dict",data_path)
        source_dictionary = Dictionary.load(str(data_path.parent / "dict.txt"))


        assert len(task_SMILES) == len(
            dataset
        ), f"Real and filtered dataset {task} do not have same length."

        text = [canonize_smile(smile) for smile in task_SMILES]
        #print('text',text)
        
        tokenizer = None
        embeds.append(
            compute_model_output(
                dataset,
                model,
                text,
                source_dictionary,
                False,
                False,
                True, #true for embeddings
                True, #true for eos_embeddings
                tokenizer,
            )[2]
        )
   # print("attention encodings",len(attention_encodings[0]))
   # print(len(attention_encodings))
    output = list(zip(*embeds))
    labels = np.array(task_labels).transpose()[0]
    #print("labels",labels)
    #print(len(labels))
    return embeds

def get_clean_embeds(embeds,failedSmiPos,posToKeep_list):
    """Clean emeddings of embeddings that encode information for digitis or hydrogens

    Args:
        embeds (_List[List[float]_): Embeddings of a SMILES
        failedSmiPos (_list_): Positions of SMILES in list where no file and/or assignment could be generated
        posToKeep_list (_list_): List of positions in a SMILES according to tokens that need to be kept (not digits or hydrogens)

    Returns:
        _list[float]_: Embeddings that do not encode hydrogens or digits, but only atoms
    """
    #throw out all pos where smiles could not be translated into atom type assignments
    embeds_clean = list()
    for count,emb in enumerate(embeds[0]):
        if count not in failedSmiPos: #assignment for this SMILES did not fail
            embeds_clean.append(emb)
    print(f"Length embeddings before removal: {len(embeds[0])}, after removal where atom assignment failed: {len(embeds_clean)}")
    assert creation_assignment_fail == (len(embeds[0])-len(embeds_clean)), f"Assignment fails ({creation_assignment_fail}) and number of deleted embeddings do not agree ({(len(embeds[0])-len(embeds_clean))})."

    #print(embeds_clean)
    #within embeddings throw out all embeddings that belong to structural tokens etc according to posToKeep_list
    embeds_cleaner = []
    assert len(embeds_clean)==(len(posToKeep_list)), f"Not the same amount of embeddings as assigned SMILES. {len(embeds_clean)} embeddings vs. {(len(posToKeep_list))} SMILES with positions"
    for smiemb,pos_list in zip(embeds_clean,posToKeep_list):
        newembsforsmi = []
        #check that atom assignment is not null
        newembsforsmi = [smiemb[pos] for pos in pos_list]
        embeds_cleaner.append(newembsforsmi)
        
    for smiemb, pos_list in zip(embeds_cleaner,posToKeep_list):
        assert len(smiemb)==len(pos_list), "Final selected embeddings for assigned atoms do not have same length as list of assigned atoms."
    return embeds_cleaner

def correctLengths(smi_toks,embeds):
    """Check that number of tokens corresponds to number of embeddings, otherwise sth went wrong

    Args:
        smi_toks (_list[string]_): SMILES tokens for a SMILES
        embeds (_list[float]_): Embeddings
    """
        #iterate through smiles and embeds compare lens
    samenums=0
    diffnums=0
    smismaller=0
    new_embs=list() #only embeds that have same length as smiles
    for smi,embs in zip(smi_toks,embeds[0]):
        if len(smi)==len(embs):
            samenums+=1
            new_embs.append(embs)
        else:
            print(f"smilen: {len(smi)} emblen: {len(embs)}")
            print(f"{smi} and len diff {len(smi)-len(embs)}")
            diffnums+=1
            if len(smi)<len(embs):
                smismaller+=1
    if diffnums==0:
        return True
    else:
        print(f"samenums: {samenums} and diffnums: {diffnums} of which smiles have smaller length: {smismaller}")
        perc = (diffnums/(diffnums+samenums))*100
        print("percentage of embeddings not correct compared to smiles: {:.2f}".format(perc))
        return False


def link_embeds_to_atomassigns(embeds_clean,smiToAtomAssign_dict_clean):
    """Linking the embeddings to their atom assignments

    Args:
        embeds_clean (list[float]): Embeddings that do not encode hydrogens or digits, but only atoms
        smiToAtomAssign_dict_clean (dict): Dictionary that links SMILES to their atom assignments

    Returns:
        _dict_: Dictionary that links SMILES to their corresponding embeddings and assignmnets
    """
    #dikt_clean[smi] = (posToKeep,smi_clean,atoms_ass_list)
    #dict[smiles]=(embedding of SMILES, assigned atoms)
    embass_dikt = dict()
    assert (len(smiToAtomAssign_dict_clean.keys())==(len(embeds_clean))), f"Number of assigned SMILES ({len(smiToAtomAssign_dict_clean.keys())}) and embeddings {len(embeds_clean)} do not agree."
    it = 0
    for smi, value in smiToAtomAssign_dict_clean.items():
        clean_toks=value[1]
        assigns = value[2]
        #print("SMILES: ",smi)
        #print(f"{clean_toks}")
        assert len(clean_toks)==(len(embeds_clean[it])), f"Number of tokens ({len(clean_toks)}) does not equal number of embeddings ({len(embeds_clean[it])}) for this SMILES string"
        assert len(assigns)==(len(embeds_clean[it])), f"Number of assignments ({len(assigns)}) does not equal number of embeddings ({len(embeds_clean[it])}) for this SMILES string.\n Assigns: {assigns} vs. Embeddings"
        embass_dikt[smi]=(embeds_clean[it],assigns)
        #print(f"final link: {smi}: {clean_toks} \n {assigns} embedding")
        it+=1
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
    fig,ax = plt.subplots(1)
    colours = [colour for colour in colours_dict.values()] #all colours used
    labels_tocols = [lab for lab in colours_dict.keys() ]
   
    scatterplot = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], marker='.', alpha=alpha, c=[colours_dict[x] for x in labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.13, 0.5), fontsize=8)
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
    os.makedirs(save_path.parent, exist_ok=True)
    pca = PCA(n_components=2, random_state=SEED + 6541)
    pca_embeddings = pca.fit_transform(embeddings)
    logging.info(
        f"{save_path} has the explained variance of {pca.explained_variance_ratio_}"
    )
    fig,ax = plt.subplots(1)
    ax.scatter(pca_embeddings[:, 0],pca_embeddings[:, 1],marker='.',alpha=alpha,c=[colours_dict[x] for x in labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.13, 0.5), fontsize=8)
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
    cmap = mpl.cm.get_cmap('viridis')
    nums = np.linspace(0,1.0,(len(big_set)))
    colors_vir = [cmap(num) for num in nums]
    # https://sashamaps.net/docs/resources/20-colors/
    colors_sash = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']
    
    #create atomtypetocolor_dict
    atomtype2color = dict()
    #sort big set according to alphabet to keep carbons closer together
    set_list = sorted(list(big_set))
    #print(f"set_before sorting: {big_set}")
    #print(f"sorted big set: {set_list}")
    
    for atype, col in zip(set_list,colors_sash[0:len(set_list)]):
        atomtype2color[atype]=col
    
    #for atype, col in zip(set_list,colors_vir):
     #   if atype=="cl":
     #       atomtype2color[atype]='#e6194B'
     #   elif atype=="f":
     #       atomtype2color[atype]='#f58231'
     #   elif atype=="o":
     #       atomtype2color[atype]='#f032e6'
     #   elif atype=="OS":
     #       atomtype2color[atype]='#dcbeff'
     #   elif atype=="DU":
     #       atomtype2color[atype]='#42d4f4'
     #   else:
     #       atomtype2color[atype]=col
    #print(atomtype2color)
    return atomtype2color, set_list

def create_elementsubsets(big_set,embeds_fin_singlelist,atom_assigns_fin_singlelist):
    """Creation of element subsets according to alphabet

    Args:
        big_set (_set_): Set of atom types
        embeds_fin_singlelist (_list[float]_): List of embeddings
        atom_assigns_fin_singlelist (_list[string]_): List of atom assignments

    Returns:
        _list,dict[string][list[float],list[string]]_: List of keys (elements), dictionary that contains ambeddings and their atomtypes sorted by element
    """
    dikt = dict()
    last_firstval = ''
    curr_liste = list()
    ctr=0
    print(big_set)
    for s in big_set:
        if ctr==0:
            last_firstval=s[0]
        if last_firstval!=s[0] or ctr==len(big_set)-1 and s!='cl': #cl will be treated differently
            dikt[last_firstval]=curr_liste
            curr_liste=list()
            last_firstval=s[0]
        if last_firstval==s[0]:
            if s!='cl': #cl will be treated differently
                curr_liste.append(s)   
        ctr+=1  
    dikt['cl']=['cl'] #cl
    print(dikt.items())
         
    it=0
    keylist = dikt.keys()
    print("keylist",keylist)
    dikt_forelems = dict()
    for elem in keylist:
        curr_emblist = list()
        curr_asslist = list()
        for ass, emb in zip(atom_assigns_fin_singlelist,embeds_fin_singlelist):
            if elem==ass[0] and ass!='cl':
                #print(f"elem {elem} equals assignment {ass}")
                curr_emblist.append(emb)
                curr_asslist.append(ass)
        dikt_forelems[elem]=(curr_emblist,curr_asslist)
    curr_emblist = list()
    curr_asslist = list()   
    for ass, emb in zip(atom_assigns_fin_singlelist,embeds_fin_singlelist):
        if ass=='cl':
            print("ass is cl",ass)
            #print(f"elem {elem} equals assignment {ass}")
            curr_emblist.append(emb)
            curr_asslist.append(ass)
    dikt_forelems['cl']=(curr_emblist,curr_asslist)
    print("dikt for elems keys:",dikt_forelems.keys())
    assert len(keylist)==(len(dikt_forelems.keys())), "Keylist and list of elements in dict not the same."
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
    p_f_list_embs = (dikt_forelems['p'][0]) + dikt_forelems['f'][0]
    #print("dikt for elems cl",(dikt_forelems['cl'][0]))
    p_f_cl_list_embs = p_f_list_embs + (dikt_forelems['cl'][0])
    p_f_list_assigs = (dikt_forelems['p'][1]) + dikt_forelems['f'][1]
    p_f_cl_list_assigs = p_f_list_assigs + (dikt_forelems['cl'][1])
    assert len(p_f_cl_list_embs)==len(p_f_cl_list_assigs)
    print("assiglist",p_f_cl_list_assigs)
    atomtype2color, set_list = getcolorstoatomtype(set(p_f_cl_list_assigs))
    pathway=Path(str(save_path_prefix)+ f"{min_dist}_{n_neighbors}_pfcl.svg")
    plot_umap(p_f_cl_list_embs, p_f_cl_list_assigs, atomtype2color, set_list, pathway, min_dist, n_neighbors, alpha)
    
    
    for key in keylist:
        print(key)
        pathway_umap=Path(str(save_path_prefix)+ f"{min_dist}_{n_neighbors}_{key}.svg")
        pathway_pca=Path(str(save_path_prefix)+ f"pca_{key}.svg")
        embeddings = dikt_forelems[key][0]
        assignments = dikt_forelems[key][1]
        atomtype2color, set_list = getcolorstoatomtype(set(assignments))
        assert len(embeddings)==(len(assignments)), "Assignments and embeddings do not have same length."
        plot_umap(embeddings, assignments, atomtype2color, set_list, pathway, min_dist, n_neighbors, alpha)
        plot_pca(embeddings, assignments, atomtype2color, pathway_pca, alpha)
    pathway_pca=Path(str(save_path_prefix)+ f"pca_pfcl.svg")
    plot_pca(p_f_cl_list_embs, p_f_cl_list_assigs, atomtype2color,pathway_pca,alpha)

if __name__ == "__main__":
    
    #get SMILES from delaney
    task = "delaney"
    #task = "bace_classification"
    assert task in list(
        MOLNET_DIRECTORY.keys()
    ), f"{task} not in MOLNET tasks."
    task_SMILES, task_labels = load_molnet_test_set(task)
    print(f"SMILES: {task_SMILES} \n len task_SMILES delaney: {len(task_SMILES)}")
    #print(f"task labels",task_labels)
    rndm_smiles = task_SMILES
    print(f"first smiles {task_SMILES[0]} and length {len(task_SMILES[0])}")
    
    #get tokenized version of dataset
    tokenizer = get_tokenizer(TOKENIZER_PATH)
    smi_toks = tokenize_dataset(tokenizer, task_SMILES, False)
    smi_toks = [smi_tok.split() for smi_tok in smi_toks]
    
    #get atom assignments from SMILES
    #smiToAtomAssign_dict, smiToAtomAssign_dict_clean, posToKeep_list, creation_assignment_fail, failedSmiPos, cleanSmis = get_atom_assignments(task_SMILES,smi_toks)
    
    #get atom assignments from folder
    smiToAtomAssign_dict, smiToAtomAssign_dict_clean, posToKeep_list, creation_assignment_fail, failedSmiPos, cleanSmis = load_assignments_from_folder("./delaney_mols_bccc0_gaff2_assigned", task_SMILES, smi_toks)

    #get embeddings per token
    embeds = []
    embeds = get_embeddings(task, False)
    
    #check that attention encodings as long as keys in dict
    assert len(smiToAtomAssign_dict.keys())==(len(embeds[0])), f"Number of SMILES and embeddings do not agree. Number of SMILES: {len(smiToAtomAssign_dict.keys())} and Number of embeddings: {len(embeds[0])}"
    assert correctLengths(smi_toks,embeds)
    embeds_clean = get_clean_embeds(embeds,failedSmiPos,posToKeep_list)
    
    embass_dikt = link_embeds_to_atomassigns(embeds_clean,smiToAtomAssign_dict_clean)
    
    embeds_fin = [val[0] for val in embass_dikt.values()]
    atom_assigns_fin = [val[1] for val in embass_dikt.values()]
    mol_labels = [num for num in range(0,len(embeds_fin))]
    
    #extract embeddings without atoms from embeds_fin 
    for emb, ass in zip(embeds_fin,atom_assigns_fin):
        assert len(emb)==(len(ass)), f"embeddings for smi and assignments do not have same length: {len(emb)} vs {len(ass)}"
    
    embeds_fin_singlelist = list()
    for smiembed in embeds_fin:
        for atomembed in smiembed:
            embeds_fin_singlelist.append(atomembed[0])
    print(f"final list of embeddings/points: {len(embeds_fin_singlelist)}")
    
    atom_assigns_fin_singlelist = list()
    for smiassigns in atom_assigns_fin:
        for singleatomassign in smiassigns:
            atom_assigns_fin_singlelist.append(singleatomassign)
    print(f"final list of assignments: {len(atom_assigns_fin_singlelist)}")
    
    #create a set from atom types for each list, create a a greater set from it, assign each atom type a color
    atomtype_set = [set(type_list) for type_list in atom_assigns_fin]
    big_set = set().union(*atomtype_set)
    big_set = sorted(list(big_set))
    print(big_set)
    
    #for atoms in big set create separate embedding lists
    keylist, dikt_forelems = create_elementsubsets(big_set,embeds_fin_singlelist,atom_assigns_fin_singlelist)
    
    min_dist = 0.1
    n_neighbors = 15
    alpha = 0.2
    save_path_prefix = f"plots/embeddingsvsatomtype/{task}"
    create_plotsperelem(keylist,dikt_forelems,min_dist,n_neighbors,alpha,save_path_prefix)
    
    atomtype2color, set_list = getcolorstoatomtype(big_set)
    

    #plot_umap(embeds_fin_singlelist, atom_assigns_fin_singlelist, atomtype2color, set_list, Path(str(save_path_prefix) + f"{min_dist}_{n_neighbors}_umap_fromfolder.svg"), min_dist, n_neighbors, alpha)