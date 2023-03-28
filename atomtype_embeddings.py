import pandas as pd
import numpy as np
import os
import selfies
from typing import List, Tuple
from io import StringIO
import logging

from correlate_embeddings import sample_random_molecules
from fairseq_utils import compute_model_output, load_dataset
from preprocessing import translate_selfie
from fairseq.data import Dictionary
from scoring import load_model
from attention_readout import load_molnet_test_set, canonize_smile

from constants import (
    TASK_MODEL_PATH,
    TASK_PATH,
    PROJECT_PATH,
    MOLNET_DIRECTORY
)

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
    struc_toks=r"()=:~1234567890#"
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
        print("##############################################################################################################")
        print(f"SMILES: {smi}")
        smi_clean, posToKeep = clean_SMILES(smi)
        smi_fi = smilestofile(smi,no,"pdb")
        if os.path.isfile(smi_fi)==True:
            print("Successful conversion of SMILES to file")
            smi_ac = exec_antechamber(smi_fi,"pdb")
            if os.path.isfile(smi_ac)==True:
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
    return dikt, dikt_clean, posToKeep_list, filecreation_fail, assignment_fail, failedSmiPos


def generate_attention_dict2(task: str, cuda: int
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
            / encoding
            / "5e-05_0.2_based_norm"
            / "5e-05_0.2_based_norm"
            / "checkpoint_best.pt"
        )
        data_path = TASK_PATH / task / encoding
        dataset = load_dataset(data_path, False)
        print(dataset)
        model = load_model(specific_model_path, data_path, cuda)
        model.zero_grad()
        data_path = data_path / "input0" / "test"

        source_dictionary = Dictionary.load(str(data_path.parent / "dict.txt"))

        assert len(task_SMILES) == len(
            dataset
        ), f"Real and filtered dataset {task} do not have same length."

        text = [canonize_smile(smile) for smile in task_SMILES]
        #if encoding.startswith("selfies"):
        #    text = [translate_selfie(smile)[0] for smile in text]

        #if encoding.endswith("sentencepiece"):
        #    tokenizer = get_tokenizer(TOKENIZER_PATH / encoding)
        #else:
        tokenizer = None
        attention_encodings.append(
            compute_model_output(
                dataset,
                model,
                text,
                source_dictionary,
                False,
                True,
                False, #true for embeddings
                False,
                tokenizer,
            )[1]
        )
    print(attention_encodings)
    output = list(zip(*attention_encodings))
    labels = np.array(task_labels).transpose()[0]
    return output, labels



def get_embedding_outputs_old(rndm_smiles,task):
    #compute_embedding_output(
    #dataset: List[np.ndarray], model, text: List[str], source_dictionary, tokenizer=None) -> List[List[Tuple[float, str]]]:
    
    for encoding in [
        "smiles_atom"#,
        #"selfies_atom",
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
        cuda = False
        model = load_model(specific_model_path, data_path, cuda)
        model.zero_grad()
        dataset = load_dataset(data_path, False) #false since regression, remark: loading dataset before atting input0/test since this is a regression task
        print("the dataset: ",dataset)
        data_path = data_path / "input0" / "test"

        source_dictionary = Dictionary.load(str(data_path.parent / "dict.txt"))
        print(source_dictionary)
        
        text = [canonize_smile(smile) for smile in task_SMILES]
        if encoding.startswith("selfies"):
            text = [translate_selfie(smile)[0] for smile in text]

       # if encoding.endswith("sentencepiece"):
       #     tokenizer = get_tokenizer(TOKENIZER_PATH / encoding)
       # else:
        tokenizer = None
            
        attention_encodings.append(
            compute_model_output(
                dataset,
                model,
                text,
                source_dictionary,
                False,
                False,
                True, #true for embeddings
                False,
                tokenizer,
            )[2] #2 for dataset_embeddings
        )
    return attention_encodings
    


if __name__ == "__main__":

    testmol = "[N-]=[P-][C-][H]"
    #get random smiles
    rndm_mols = sample_random_molecules(1)
    rndm_smiles = rndm_mols.loc[:,"rnd_smiles"]
    rndm_smiles_np = rndm_mols.loc[:,"rnd_smiles"].to_numpy()
    rndm_smiles_new = [smi for smi in rndm_smiles]
    print(rndm_smiles_new)
    print(rndm_smiles_np)
    
    #get SMILES from delaney
    task = "delaney"
    assert task in list(
        MOLNET_DIRECTORY.keys()
    ), f"{task} not in MOLNET tasks."
    task_SMILES, task_labels = load_molnet_test_set(task)
    print(f"SMILES: {task_SMILES} \n len task_SMILES delaney: {len(task_SMILES)}")
    rndm_smiles = task_SMILES
    
    #get atom assignments
    smiToAtomAssign_dict, smiToAtomAssign_dict_clean, posToKeep_list, filecreation_fail, assignment_fail,failedSmiPos = get_atom_assignments(task_SMILES)
    ##smiatomassign_dict as long as input smiles array
    print(smiToAtomAssign_dict)
    
    #print number of fails
    logging.info(f"File creation from SMILES to pdb by obabel failed {filecreation_fail} times out of {len(rndm_smiles)}")
    logging.info(f"Atom assignment by antechamber failed {assignment_fail} times out of {len(rndm_smiles)}")
    #print(f"File creation from SMILES to pdb by obable failed {filecreation_fail} times out {len(rndm_smiles)}")
    #print(f"Atom assignment by antechamber failed {assignment_fail} times out {len(rndm_smiles)}")
    
    #get embeddings per token
    attention_encodings = []

    #attention_encodings  = generate_attention_dict2(task, False)
    
    #check that attention encodings as long as keys in dict
    #assert(len(smiToAtomAssign_dict.keys())==(len(attention_encodings)))
    
    #throw out all pos where smiles could not be translated into atom type assignments
    #attention_encodings_clean = [(attention_encodings.remove(pos)) for pos in failedSmiPos]

    #within embeddings throw out all embeddings that belong to structural tokens etc accoridng to posToKeep_list
    #attention_encodings_cleaner = []
   # for smiemb,pos_list in zip(attention_encodings,posToKeep_list):
     #   newembsforsmi = []
        #check that atom assignment is not null
     #   if pos_list is not None:
     #       newembsforsmi = [smiemb[pos] for pos in pos_list]
     #       attention_encodings_cleaner.append(newembsforsmi)
    
    
