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

from correlate_embeddings import sample_random_molecules
from fairseq_utils import compute_model_output, load_dataset
from preprocessing import translate_selfie
from fairseq.data import Dictionary
from scoring import load_model
from attention_readout import load_molnet_test_set, canonize_smile
from plotting import plot_representations
from tokenisation import tokenize_dataset, get_tokenizer
from constants import SEED

from constants import (
    TASK_MODEL_PATH,
    TASK_PATH,
    MOLNET_DIRECTORY,
    TOKENIZER_PATH
)
markers = list(Line2D.markers.keys())
prop_cycle = plt.rcParams["axes.prop_cycle"]
default_colours = prop_cycle.by_key()["color"]

#translate SMILES to mol2 (obabel)
def smilestofile(smiles,no,ftype):
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
    inputfile_noex=os.path.splitext(inputfile)[0]
    #print("inputfile no extension",inputfile_noex)
    if ftype=="pdb":
        #os.system(f"antechamber -i {inputfile} -fi pdb -o {inputfile_noex}_ass.mol2 -fo mol2 -c bcc -nc 0 -at gaff2")
        os.system(f"antechamber -i {inputfile} -fi pdb -o {inputfile_noex}_ass.mol2 -fo mol2 -at gaff2")
    elif ftype=="mol2":
        os.system(f"antechamber -i {inputfile} -fi mol2 -o {inputfile_noex}_ass.mol2 -fo mol2 -c bcc -nc 0 -at gaff2")
    else:
        print("Execution of antechamber failed. Wrong filetype given. Filetype needs to be pdb or mol2.")
        return None
    return f"{inputfile_noex}_ass.mol2"

def run_parmchk2(ac_outfile):
    acout_noex=os.path.splitext(ac_outfile)[0]
    print("acout_noex",acout_noex)
    os.system(f"parmchk2 -i {ac_outfile} -f mol2 -o {acout_noex}.frcmod -s gaff2")
    #check whether file was generated
    if os.path.isfile(f"{acout_noex}.frcmod")==True:
        with open(f"{acout_noex}.frcmod") as infile:
            lines = infile.read().splitlines()
            for line in lines:
                if "ATTN: needs revision" in line:
                    print("###############################################################################")
                    print("###############################################################################")
                    print("##########################################ATTENTION###############################")
                    print("###############################################################################")
                    print("###############################################################################")
                    return False
    return True

#get atom assignment
def getatom_ass(mol2):
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
    atoms_ass_set = set(atoms_ass_list)
    return atoms_ass_list, atoms_ass_set

def clean_acout(ac_out) -> list:
    ac_out_noH=list()
    for j in ac_out:
        #save only when it's mot H
        if not j.startswith('H'):
            ac_out_noH.append(j)
            #print(f"-----------------this is not H, this is: {j}")
    #print("before: ", ac_out)
    #print("after: ", ac_out_noH)
    return ac_out_noH

def clean_SMILES(SMILES_tok):
    SMILES_tok_prep=list()
    struc_toks=r"()=:~1234567890#"
    posToKeep=list()
    pos=0
    for i in range(len(SMILES_tok)):
        #when it's an H in the SMILES, ignore, cannot deal
        #print(SMILES_tok[i])
        if SMILES_tok[i]!="H" and not SMILES_tok[i].isdigit() and not SMILES_tok[i].isspace():
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

def get_atom_assignments(smiles_arr,smi_toks):
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
        smi_fi = smilestofile(smi,no,"pdb")
        if os.path.isfile(smi_fi)==True:
            print("Successful conversion of SMILES to file")
            smi_ac = exec_antechamber(smi_fi,"pdb")
            if os.path.isfile(smi_ac)==True:
                #if smi_ac was generated check if with parmchk2, returns True if output is ok
                if True==run_parmchk2(smi_ac):
                    #get antechamber assignment
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
        assert len(assigns)==(len(embeds_clean[it])), f"Number of assignments ({len(assigns)}) does not equal number of embeddings ({len(embeds_clean[it])}) for this SMILES string"
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
    logging.info("Started plotting UMAP")
    os.makedirs(save_path.parent, exist_ok=True)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=SEED + 6539
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    
    
    fig,ax = plt.subplots(1)
    colours = [colour for colour in colours_dict.values()] #all colours used
    labels_tocols = [lab for lab in colours_dict.keys() ]
   # plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap=[colours[x] for x in labels])
    # only plot carbons
    #for label,umap_emb in zip(labels,umap_embeddings):
    #    if label.startswith("C") and label!="Cl":
    #        ax.scatter(umap_emb[0],umap_emb[1], marker='.', c=[colours_dict[x] for x in labels])
   
    scatterplot = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], marker='.', c=[colours_dict[x] for x in labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.13, 0.5), fontsize=8)
    ax.set_ylabel("UMAP 2")
    ax.set_xlabel("UMAP 1")
    ax.set_title("UMAP - Embeddings resp. to atom types")
    fig.savefig(save_path, format="svg")
    fig.clf()

        
def getcolorstoatomtype(big_set):
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
    
    for atype, col in zip(set_list,colors_vir):
        if atype=="Cl":
            atomtype2color[atype]='#e6194B'
        elif atype=="F":
            atomtype2color[atype]='#f58231'
        elif atype=="O":
            atomtype2color[atype]='#f032e6'
        elif atype=="OS":
            atomtype2color[atype]='#dcbeff'
        elif atype=="DU":
            atomtype2color[atype]='#42d4f4'
        else:
            atomtype2color[atype]=col
    #print(atomtype2color)
    return atomtype2color, set_list

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
    #smi_toks = [smi_tok.replace(" ","") for smi_tok in smi_toks]
    smi_toks = [smi_tok.split() for smi_tok in smi_toks]
    #get atom assignments
    smiToAtomAssign_dict, smiToAtomAssign_dict_clean, posToKeep_list, creation_assignment_fail, failedSmiPos, cleanSmis = get_atom_assignments(task_SMILES,smi_toks)
    ##smiatomassign_dict as long as input smiles array
    #print("[][]][][][][][][][][][][][][][][][][][][][][]Atom assignment dict[][]][][][][][][][][][][][][][][][][][][][][]")
    #print(smiToAtomAssign_dict)
    #print("[][]][][][][][][][][][][][][][][][][][][][][][][][]][][][][][][][][][]][][][][][][][][][][][][][][][][][][][][]")

    #get embeddings per token
    embeds = []
    embeds = get_embeddings(task, False)
    
    #check that attention encodings as long as keys in dict
    assert len(smiToAtomAssign_dict.keys())==(len(embeds[0])), f"Number of SMILES and embeddings do not agree. Number of SMILES: {len(smiToAtomAssign_dict.keys())} and Number of embeddings: {len(embeds[0])}"
    assert correctLengths(smi_toks,embeds)
    embeds_clean = get_clean_embeds(embeds,failedSmiPos,posToKeep_list)
    
    embass_dikt = link_embeds_to_atomassigns(embeds_clean,smiToAtomAssign_dict_clean)
    
    #from 112 pick 30 first and only embedding not description of atom
    embeds_fin = [val[0] for val in embass_dikt.values()]
    atom_assigns_fin = [val[1] for val in embass_dikt.values()]
    mol_labels = [num for num in range(0,len(embeds_fin))]
    
    #extract embeddings without atoms from  embeds_fin 
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
    print(f"final list of assignmnets: {len(atom_assigns_fin_singlelist)}")
    
    #create a set from atom types for each list, create a a greater set from it, assign each atom type a color
    atomtype_set = [set(type_list) for type_list in atom_assigns_fin]
    big_set = set().union(*atomtype_set)
    
    atomtype2color, set_list = getcolorstoatomtype(big_set)
    
    min_dist = 0.3
    n_neighbors = 15
    alpha = 0.2
    save_path_prefix = f"plots/embeddingsvsatomtype/{task}"
    plot_umap(embeds_fin_singlelist, atom_assigns_fin_singlelist, atomtype2color, set_list, Path(str(save_path_prefix) + f"{min_dist}_{n_neighbors}_umap_discol_gaff2.svg"), min_dist, n_neighbors, alpha)