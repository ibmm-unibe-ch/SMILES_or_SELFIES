# File to get embeddings for pretrained dataset for 
# SMILES
# SELFIES
# BART
# RoBERTa
# use env fairseq_git2
import os, sys
import json
import logging
from typing import List
from tokenisation import tokenize_dataset, get_tokenizer
from pathlib import Path
from fairseq_utils2 import compute_model_output, compute_model_output_RoBERTa, compute_random_model_output, load_dataset, load_model
from fairseq.data import Dictionary
from SMILES_to_SELFIES_mapping import canonize_smiles, generate_mapping, generate_mappings_for_task_SMILES_to_SELFIES
from itertools import chain
from constants import SEED
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from collections import Counter
import pandas as pd

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
            f"same numbers between tokens and embeddings: {samenums} and different number betqween tokens and embeddings: {diffnums} of which smiles tokens have smaller length: {smismaller}")
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
    assert len(task_reps) == len(
        dataset
    ), f"Real and filtered dataset {task} do not have same length: len(task_reps): {len(task_reps)} vs. len(dataset):{len(dataset)} ."
    

    #text = [canonize_smile(smile) for smile in task_SMILES]
    text = [rep for rep in task_reps]
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
    elif rep=="selfies":
        #subfolder="selfies_atom_isomers"
        subfolder="selfies_atom_standard"
        
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
    data_path = Path("/scratch/ifender/SOS_tmp/")
    #data_path = TASK_PATH/"bbbp"/f"{subfolder}"
    #fairseq_dict = Dictionary.load(str(fairseq_dict_path))
    #fairseq_dict_path = FAIRSEQ_PREPROCESS_PATH/ "smiles_atom_isomers"/"dict.txt"
    
    embeds = []
    embeds = get_embeddings(task, specific_model_path, data_path, False, reps) #works for BART model with newest version of fairseq on github, see fairseq_git.yaml file
    checked_embeds = check_lengths(listoftokenisedreps, embeds) #, "Length of SMILES_tokens and embeddings do not agree."
    print("got the embeddings")
    return checked_embeds


if __name__ == "__main__":

    # all SMILES from pretrained dataset, yes you really need to get all because that is how the get embeddings function works
    csv = '/data/jgut/SMILES_or_SELFIES/processed/isomers/full_deduplicated_isomers.csv'
    df = pd.read_csv(csv)
    task_SMILES = df['SMILES'].tolist()
    #print('Canonizing SMILES')
    task_SMILES = [canonize_smiles(smiles) for smiles in task_SMILES]
        
    tokenizer = get_tokenizer(TOKENIZER_PATH)
    #print(f"tokenizer {tokenizer}")
    smi_toks = tokenize_dataset(tokenizer, task_SMILES, False)
    #print("whole SMILES tokenized: ",smi_toks[0])
    smi_toks = [smi_tok.split() for smi_tok in smi_toks]
    #print(f"SMILES tokens after splitting tokens into single strings: {smi_toks[0]}")
    smiles_dict = dict(zip(task_SMILES,smi_toks))

    #for k, v in smiles_dict.items():
    #    print(f"SMILES: {k}")
    #    print(f"Tokens: {v}")


    ## map SMILES to SELFIES and get SELFIES and SELFIES tokens
    
    smiles_to_selfies_mapping = generate_mappings_for_task_SMILES_to_SELFIES(task_SMILES)
    selfies_tokenised = []
    selfies = []
    maps_num = 0
    for key in smiles_to_selfies_mapping.keys():
        #print(f"SMILES: {key} SELFIES: {smiles_to_selfies_mapping[key]}")
        selfies_tokenised.append(smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][1])
        selfies.append(smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][0])
    
    
    #####################################get actual embeddings for 4 models
    # task can be anything, 
    task = 'pretrained'
    # traintype choose pretrained, 
    traintype="pretrained"
    
    outfolder = "/scratch/ifender/SOS_tmp/embeddings_pretrainingdata"
    os.makedirs(outfolder, exist_ok=True)

    configs = [
        {"model": "BART", "rep": "smiles", "keys": list(smiles_dict.keys()), "values": list(smiles_dict.values())},
        {"model": "roberta", "rep": "smiles", "keys": list(smiles_dict.keys()), "values": list(smiles_dict.values())},
        {"model": "BART", "rep": "selfies", "keys": selfies, "values": selfies_tokenised},
        {"model": "roberta", "rep": "selfies", "keys": selfies, "values": selfies_tokenised},
    ]

    for cfg in configs:
        try:
            print(f"Getting embeddings for model={cfg['model']} rep={cfg['rep']}")
            embeds = get_embeddings_from_model(
                task,
                traintype,
                cfg["model"],
                cfg["rep"],
                cfg["keys"],
                cfg["values"],
            )
            df = pd.DataFrame({
                cfg["rep"].upper(): cfg["keys"],
                "embedding": embeds[0]  # adjust if needed
            })
            df.to_csv(f"{outfolder}/embeds_{cfg['rep']}_{task}_{cfg['model']}.csv", index=False)
        except Exception as e:
            print(f"Error occurred while getting embeddings for model={cfg['model']} rep={cfg['rep']}: {e}")
