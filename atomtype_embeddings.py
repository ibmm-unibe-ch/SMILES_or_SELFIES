import pandas as pd
import numpy as np
import os
from typing import List
from io import StringIO
from correlate_embeddings import sample_random_molecules,sample_synonym

from fairseq_utils import compute_embedding_output, compute_model_output
from preprocessing import translate_selfie
from tokenisation import get_tokenizer
from fairseq.data import Dictionary
from scoring import load_dataset, load_model


from constants import (
    MOLNET_DIRECTORY,
    PARSING_REGEX,
    TASK_MODEL_PATH,
    TASK_PATH,
    TOKENIZER_PATH,
)
from constants import PARSING_REGEX, PROJECT_PATH, TASK_PATH

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
    print("inputfile no extension",inputfile_noex)
    if ftype=="pdb":
        os.system(f"antechamber -i {inputfile} -fi pdb -o {inputfile_noex}_ass.mol2 -fo mol2 -at amber")
    elif ftype=="mol2":
        os.system(f"antechamber -i {inputfile} -fi mol2 -o {inputfile_noex}_ass.mol2 -fo mol2 -at amber")
    else:
        print("Execution of antechamber failed. Wrong filetype given. Filetype needs to be pdb or mol2.")
        return None
    return f"{inputfile_noex}_ass.mol2"

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
    #print("atoms assigned", atoms_ass_list)
    #print()
    #print("assignments set", atoms_ass_set)
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
    struc_toks=r"()=-:~1234567890#"
    posToKeep=list()
    pos=0
    for i in range(len(SMILES_tok)):
        #when it's an H in the SMILES, ignore, cannot deal
        #print(SMILES_tok[i])
        if SMILES_tok[i]!="H" and not SMILES_tok[i].isdigit():
            if any(elem in struc_toks for elem in SMILES_tok[i])==False:
                SMILES_tok_prep.append(SMILES_tok[i])
                posToKeep.append(pos) #keep pos where you keep SMILES token
        pos+=1
    assert(len(posToKeep)==(len(SMILES_tok_prep)))
    return SMILES_tok_prep,posToKeep

def get_atom_assignments(smiles_arr):
    no=0
    assignment_list=list()
    dikt=dict()
    dikt_clean = dict()
    posToKeep_list = list()
    filecreation_fail = 0
    assignment_fail = 0
    smi_num = 0
    failedSmiPos = list()
    for smi in smiles_arr:
        smi_clean, posToKeep = clean_SMILES(smi)
        smi_fi = smilestofile(smi,no,"pdb")
        if smilestofile is not None:
            smi_ac = exec_antechamber(smi_fi,"pdb")
            if smi_ac is not None:
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
    assert(len(dic.keys())==(len(smiles_arr)))
    return dikt, dikt_clean, posToKeep_list, filecreation_fail, assignment_fail, failedSmiPos

def get_embedding_outputs(rndm_smiles,task) -> Any:
    #compute_embedding_output(
    #dataset: List[np.ndarray], model, text: List[str], source_dictionary, tokenizer=None) -> List[List[Tuple[float, str]]]:
    
    for encoding in [
        "smiles_atom",
        "selfies_atom",
    ]:
        specific_model_path = (
            TASK_MODEL_PATH
            / task
            / encoding
            / "5e-05_0.2_based_norm"
            / "5e-05_0.2_based_norm"
            / "checkpoint_best.pt"
        )
        data_path = TASK_PATH / task / encoding
        model = load_model(specific_model_path, data_path, cuda)
        model.zero_grad()
        data_path = data_path / "input0" / "test"
        source_dictionary = Dictionary.load(str(data_path.parent / "dict.txt"))

        if encoding.startswith("selfies"):
            text = [translate_selfie(smile)[0] for smile in text]

        if encoding.endswith("sentencepiece"):
            tokenizer = get_tokenizer(TOKENIZER_PATH / encoding)
        else:
            tokenizer = None
            
        attention_encodings.append(
            compute_model_output(
                List[rndm_smiles.to_numpy()],
                model,
                List[rndm_smiles],
                source_dictionary,
                False,
                False,
                True, #true for embeddings
                False,
                tokenizer,
            )[2]
        )
    return attention_encodings
    


if __name__ == "__main__":
    print("test")

    #get random smiles
    rndm_mols = sample_random_molecules(100)
    rndm_smiles = rndm_mols.loc[:,"rnd_smiles"]
    rndm_smiles_np = rndm_mols.loc[:,"rnd_smiles"].to_numpy()
    
    #get atom assignments
    smiToAtomAssign_dict, smiToAtomAssign_dict_clean, posToKeep_list, filecreation_fail, assignment_fail,failedSmiPos = get_atom_assignments(rndm_smiles)
    ##smiatomassign_dict as long as input smiles array
    
    #print number of fails
    print(f"File creation from SMILES to pdb by obable failed {filecreation_fail} times out {len(rndm_smiles)}")
    print(f"Atom assignment by antechamber failed {assignment_fail} times out {len(rndm_smiles)}")
    
    #get embeddings per token
    attention_encodings = []
    task = "Classification"
    attention_encodings  = get_embedding_outputs(rndm_smiles,task)
    
    #check that attention encodings as long as keys in dict
    assert(len(smiToAtomAssign_dict.keys())==(len(attention_encodings)))
    
    #throw out all pos where smiles could not be translated into atom type assignments
    attention_encodings_clean = [(del attention_encodings[pos]) for pos in failedSmiPos]

    #within embeddings throw out all embeddings that belong to structural tokens etc accoridng to posToKeep_list
        
    attention_encodings_cleaner = []
    for smiemb,pos_list in zip(attention_encodings,posToKeep_list):
        newembsforsmi = []
        #check that atom assignment is not null
        if pos_list is not None:
            newembsforsmi = [smiemb[pos] for pos in pos_list]
            attention_encodings_cleaner.append(newembsforsmi)
    
    
