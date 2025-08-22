# File to get embeddings for pretrained dataset for 
# SMILES
# SELFIES
# BART
# RoBERTa
# use env fairseq_git2
# analogous to SMILEStoSELFIEStoatomtypes.ipynb
import os, sys
import json
import logging
from typing import List
from tokenisation import tokenize_dataset, get_tokenizer
from pathlib import Path
from fairseq_utils2 import load_dataset, load_model, compute_embedding_output
from fairseq.data import Dictionary
from SMILES_to_SELFIES_mapping import canonize_smiles, generate_mapping, generate_mappings_for_task_SMILES_to_SELFIES
from itertools import chain
from constants import SEED
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from collections import Counter
import pandas as pd
import traceback
import pickle
from datetime import datetime

from constants import (
    TASK_PATH,
    MOLNET_DIRECTORY,
    TOKENIZER_PATH
)


def check_lengths(smi_toks, embeds):
    """Check that number of tokens corresponds to number of embeddings per SMILES, otherwise sth went wrong
     new: if sth went wrong turn that embedding to None and return the embeddings

    Args:
        smi_toks (_list[string]_): SMILES tokens for a SMILES
        embeds (_list[float]_): Embeddings
    """
    samenums = 0
    diffnums = 0
    smismaller = 0
    new_embs = list()
    for smi, embs in zip(smi_toks, embeds[0]):
        # only compare when both are not None)
        if embs is not None and smi is not None:
            if len(smi) == len(embs):
                samenums += 1
                new_embs.append(embs)
            else:
                print(f"smilen: {len(smi)} emblen: {len(embs)}")
                embs_signs = [emb1 for (emb0,emb1) in embs]
                print(f"smi: {smi} \nemb: {embs_signs} \nwith len diff {len(smi)-len(embs)}")
                diffnums += 1
                new_embs.append(None)
                if len(smi) < len(embs):
                    smismaller += 1
    embeds[0]=new_embs
    if diffnums == 0:
        return embeds
    else:
        print(
            f"same numbers between tokens and embeddings: {samenums} and different number between tokens and embeddings: {diffnums} of which smiles tokens have smaller length: {smismaller}")
        perc = (diffnums/(diffnums+samenums))*100
        print(
            "percentage of embeddings not correct compared to smiles: {:.2f}%".format(perc))
        return embeds

def get_embeddings(task: str, specific_model_path: str, data_path: str, cuda: int, task_reps: List[str]):
    """Generate the embeddings dict of a task
    Args:
        task (str): Task to find attention of
        cuda (int): CUDA device to use
    Returns:
        Tuple[List[List[float]], np.ndarray]: attention, labels
    """
    #task_SMILES, task_labels = load_molnet_test_set(task)
    print("in get embeddings")
    # Ensure specific_model_path and data_path are not None
    if specific_model_path is None:
        raise ValueError("specific_model_path cannot be None")
    if data_path is None:
        raise ValueError("data_path cannot be None")
    
    #data_path = "/data/jgut/SMILES_or_SELFIES/task/delaney/smiles_atom_isomers"
    print("data path: ", data_path)
    if "random" not in str(specific_model_path):
        print("loading model")
        model = load_model(specific_model_path, data_path, cuda)
        print("model loaded")
        model.zero_grad()
    data_path = data_path / "input0" / "test"
    # True for classification, false for regression
    print("loading dataset with datapath: ", data_path)
    dataset = load_dataset(data_path, True)
    print("datapath srcdict:",str(data_path.parent / "dict.txt"))
    source_dictionary = Dictionary.load(str(data_path.parent / "dict.txt"))

    # only works if run on whole dataset
    #assert len(task_reps) == len(
    #    dataset
    #), f"Real and filtered dataset {task} do not have same length: len(task_reps): {len(task_reps)} vs. len(dataset):{len(dataset)} ."
    

    #text = [canonize_smile(smile) for smile in task_SMILES]
    text = [rep for rep in task_reps]
    embeds= []
    tokenizer = None
    
    #new embedding computation
    # taken from fairseq_utils.py
    tokenizer = get_tokenizer(TOKENIZER_PATH)
    embeds.append(compute_embedding_output(model, text, source_dictionary,tokenizer))
    
    """
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
        """
   # print("attention encodings",len(attention_encodings[0]))
   # print(len(attention_encodings))
    #output = list(zip(*embeds))
    #labels = np.array(task_labels).transpose()[0]
    # print("labels",labels)
    # print(len(labels))
    return embeds

def get_embeddings_from_model(task, traintype, model, rep, reps, listoftokenisedreps):
    # ----------------------specific model paths for Delaney for BART and RoBERTa-------------------------
    finetuned_TASK_MODEL_PATH = Path("/data2/jgut/SoS_models")
    pretrained_TASK_MODEL_PATH = Path("/data/jgut/SMILES_or_SELFIES/prediction_models")
    
    # path to finetuned models
    subfolder=""
    if rep=="smiles":
        #subfolder = "smiles_atom_isomers"
        subfolder = "smiles_atom_standard"
        data_path = Path("/scratch/ifender/SOS_tmp/")
    elif rep=="selfies":
        #subfolder="selfies_atom_isomers"
        subfolder="selfies_atom_standard"
        data_path = Path("/scratch/ifender/SOS_tmp/selfies/")
        
    if model!="random":
        if traintype=="pretrained":
            if model=="BART":
                # path for BART   
                specific_model_path = (
                    pretrained_TASK_MODEL_PATH
                    / f"{subfolder}_bart"
                    / "checkpoint_last.pt"
                ) 
            else:
                #path for RoBERTa
                specific_model_path = (
                pretrained_TASK_MODEL_PATH
                / f"{subfolder}_roberta"
                / "checkpoint_last.pt"
                )
    print("specific model path: ",specific_model_path)
    
    if specific_model_path is None:
        raise ValueError("specific_model_path cannot be None")
    #data_path = TASK_PATH/"bbbp"/f"{subfolder}"
    #fairseq_dict = Dictionary.load(str(fairseq_dict_path))
    #fairseq_dict_path = FAIRSEQ_PREPROCESS_PATH/ "smiles_atom_isomers"/"dict.txt"
    
    embeds = []
    embeds = get_embeddings(task, specific_model_path, data_path, False, reps) #works for BART model with newest version of fairseq on github, see fairseq_git.yaml file
    checked_embeds = check_lengths(listoftokenisedreps, embeds) #, "Length of SMILES_tokens and embeddings do not agree."
    print("got the embeddings")
    return checked_embeds


if __name__ == "__main__":
    print(f"{datetime.now()} Starting embedding extraction")
    # 1 . Load assigned atom types
    # get the mapping SMILES to atom types from dict.json
    # Load the dictionary from the JSON file
    diktfolder = "/home/ifender/SOS/SMILES_or_SELFIES/atomtype_embedding_visualisation/assignment_dicts/dikt_pretraindataset.json"
    with open(diktfolder, 'r') as file:
        loaded_dikt = json.load(file)

    # 2. Load all SMILES 
    print("Get all SMILES from properly assigned atom type SMILES")
    task_SMILES = [smiles for smiles, value in loaded_dikt.items() if value['atom_types'] is not None]
    print(f"--Got them {len(task_SMILES)}")

    # 3. Canonize those SMILES and tell me they turn out to be the same as before
    task_SMILES_canonized = [canonize_smiles(smile) for smile in task_SMILES]
    print("Canonized SMILES:")
    #print(task_SMILES_canonized)
    # compare to task_SMILES (I already compared them in a separate notebook and they were canonized before for atom types)
    for original, canonized in zip(task_SMILES, task_SMILES_canonized):
        assert(original == canonized)
    print("                 ...no problems")

    # 4. Generate mapping between SMILES and SELFIES
    #this is what smiles_to_selfies_mapping looks like: mappings[smiles]['selfiesstr_tok_map'] = (selfies_str,tokenised_selfies,mapping)
    print("Mapping SMILES and SELFIES")
    smiles_to_selfies_mapping = generate_mappings_for_task_SMILES_to_SELFIES(task_SMILES)
    print("--Mapped")

    #5. Merge all the working atomtype mappings and smilestoselfies mappings to select only SMILES that have both informations
    print("Create dict of SMILES that have both atom types and SELFIES matching")
    smilestoatomtypestoselfies_dikt = dict()
    for smiles in task_SMILES:
        #print(smiles)
        atom_types = loaded_dikt.get(smiles, {}).get('atom_types', None)
        #print('atom types: ',atom_types)
        selfies = smiles_to_selfies_mapping.get(smiles, {}).get('selfiesstr_tok_map', (None, None, None))[0]
        selfies_toks = smiles_to_selfies_mapping.get(smiles, {}).get('selfiesstr_tok_map', (None, None, None))[1]
        selfies_map = smiles_to_selfies_mapping.get(smiles, {}).get('selfiesstr_tok_map', (None, None, None))[2]
        #print('selfies map: ',selfies_map)
        #check that neither is empty
        if selfies_map is not None and atom_types is not None: #atom types cannot be none because we filter on it, but anyway
            # final dict will have as keys to value: 'posToKeep', 'smi_clean', 'atom_types', 'max_penalty'
            smilestoatomtypestoselfies_dikt[smiles] = {**loaded_dikt.get(smiles, {}), 'selfies': selfies, 'selfies_toks': selfies_toks, 'selfies_map': selfies_map}
    #print(smilestoatomtypestoselfies_dikt)
    print(f"--Final dict of SMILES that contain info on atom types and SELFIES created and contains: {len(smilestoatomtypestoselfies_dikt)} molecules")


    # 6. Get all the corresponding SELFIES and tokenized SELFIES from the created dictionary
    print("Retrieve SELFIES and tokenized SELFIES")
    selfies_list = [v['selfies'] for v in smilestoatomtypestoselfies_dikt.values()]
    selfies_tokenised = [v['selfies_toks'] for v in smilestoatomtypestoselfies_dikt.values()]
    selfies_dict = dict(zip(selfies_list,selfies_tokenised))
    print(len(selfies_dict))

    # 7. then prepare SMILES for tokenization to get embeddings
    print("Tokenizing SMILES of pretraining set")
    tokenizer = get_tokenizer(TOKENIZER_PATH)
    #print(f"tokenizer {tokenizer}")
    smi_toks = tokenize_dataset(tokenizer, list(smilestoatomtypestoselfies_dikt.keys()), False)
    print("whole SMILES tokenized: ",smi_toks[0])
    smi_toks = [smi_tok.split() for smi_tok in smi_toks]
    #print(f"SMILES tokens after splitting tokens into single strings: {smi_toks[0]}")
    smiles_dict = dict(zip(smilestoatomtypestoselfies_dikt.keys(),smi_toks))
    print("--Created SMILES dictionary with len ", len(smiles_dict))

    assert len(smiles_dict) == len(smilestoatomtypestoselfies_dikt), "Length of SMILES dictionary and final dict do not agree."
    assert len(smiles_dict) == len(selfies_dict), "Length of SMILES dictionary and selfies do not agree."


    #for k, v in smiles_dict.items():
    #    print(f"SMILES: {k}")
    #    print(f"Tokens: {v}")
    print(f"{datetime.now()}: We now have everything needed to retrieve embeddings for SMILES, SELFIES and from both models, RoBERTa, and BART")


    #####################################get actual embeddings for 4 models
    print("#########################################Getting embeddings for filtered pretrained dataset")
    # task can be anything, 
    task = 'pretrained'
    # traintype choose pretrained, 
    traintype="pretrained"
    
    outfolder = "/scratch/ifender/SOS_tmp/embeddings_pretrainingdata"
    os.makedirs(outfolder, exist_ok=True)

    configs = [
        {"model": "BART", "rep": "smiles", "keys": list(smiles_dict.keys()), "values": list(smiles_dict.values())},
        {"model": "roberta", "rep": "smiles", "keys": list(smiles_dict.keys()), "values": list(smiles_dict.values())},
        {"model": "BART", "rep": "selfies", "keys": list(selfies_dict.keys()), "values": list(selfies_dict.values())},
        {"model": "roberta", "rep": "selfies", "keys": list(selfies_dict.keys()), "values": list(selfies_dict.values())},
    ]

    for cfg in configs:
        try:
            print(f"{datetime.now()}====Getting embeddings for model={cfg['model']} rep={cfg['rep']}==================================")
            embeds = get_embeddings_from_model(
                task,
                traintype,
                cfg["model"],
                cfg["rep"],
                cfg["keys"],
                cfg["values"],
            )
            
            # Filter out None embeddings and only save the ones that don't have none embeddings
            filtered_keys = []
            filtered_embeds = []
            for k, e in zip(cfg["keys"], embeds[0]):
                if e is not None:
                    filtered_keys.append(k)
                    filtered_embeds.append(e)
            print(f"Number of none embeddings filtered out: {len(cfg['keys']) - len(filtered_keys)}")
            print(f"Finale number of embeddings: {len(filtered_keys)}")
            
            df = pd.DataFrame({
                cfg["rep"].upper(): filtered_keys,
                "embedding": filtered_embeds
            })

            df.to_csv(f"{outfolder}/embeds_{cfg['rep']}_{task}_{cfg['model']}_22_8.csv", index=False)
            print(f"{datetime.now()}    Successfully saved embeddings for model={cfg['model']} rep={cfg['rep']}")
        except Exception as e:
            print(f"{datetime.now()}    Error occurred while getting embeddings for model={cfg['model']} rep={cfg['rep']}: {e}")
            traceback.print_exc()

    print(f"{datetime.now()} Lastly trying to save dictionary connecting SMILES to atomtypes and SMILES and SELFIES")

    with open("/scratch/ifender/SOS_tmp/embeddings_pretrainingdata/smilestoatomtypestoselfies_dikt_22_8.pkl", "wb") as f:
        pickle.dump(smilestoatomtypestoselfies_dikt, f)