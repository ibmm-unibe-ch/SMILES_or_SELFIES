"""Readout of attention
SMILES or SELFIES
"""

#from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from constants import FAIRSEQ_PREPROCESS_PATH, TOKENIZER_PATH, TASK_PATH#, MOLNET_DIRECTORY
from deepchem.feat import RawFeaturizer
from fairseq_utils import compute_attention_output, get_dictionary
from preprocessing import canonize_smile, translate_selfie
from scoring import load_model
from tokenisation import get_tokenizer



def to_markdown(molecule, smiles_atom, smiles_sentencepiece, selfies_atom, selfies_sentencepiece):
    md = ""
    md+=f'## Molecule'+"""\ """
    input_data = np.array([(letter[1], f"{letter[0]-1/len(molecule):.3f}") for letter in molecule])
    md+=f'{"".join(input_data[:,0])}'+"""\ """
    if smiles_atom:
        md+=f'## SMILES atomwise'+"""\ """
        input_data = np.array([(letter[1], f"{letter[0]-1/len(smiles_atom):.3f}") for letter in smiles_atom])
        input_data2 = np.array([(letter[1], f"{letter[0]:.3f}") for letter in smiles_atom])
        df = pd.concat([pd.DataFrame(data = input_data[:,1], index = input_data[:,0]).transpose(),pd.DataFrame(data = input_data2[:,1], index = input_data2[:,0]).transpose()])
        md += df.to_markdown()+"""\ """
    if selfies_atom:
        md+=f'## SELFIES atomwise'+"""\ """
        input_data = np.array([(letter[1], f"{letter[0]-1/len(selfies_atom):.3f}") for letter in selfies_atom])
        input_data2 = np.array([(letter[1], f"{letter[0]:.3f}") for letter in selfies_atom])
        df = pd.concat([pd.DataFrame(data = input_data[:,1], index = input_data[:,0]).transpose(),pd.DataFrame(data = input_data2[:,1], index = input_data2[:,0]).transpose()])
        md += df.to_markdown()+"""\ """
    if smiles_sentencepiece:
        md+=f'## SMILES SentencePiece'+"""\ """
        input_data = np.array([(letter[1], f"{letter[0]-1/len(smiles_sentencepiece):.3f}") for letter in smiles_sentencepiece])
        input_data2 = np.array([(letter[1], f"{letter[0]:.3f}") for letter in smiles_sentencepiece])
        df = pd.concat([pd.DataFrame(data = input_data[:,1], index = input_data[:,0]).transpose(),pd.DataFrame(data = input_data2[:,1], index = input_data2[:,0]).transpose()]) 
        md += df.to_markdown()+"""\ """
    if selfies_sentencepiece:
        md+=f'## SELFIES SentencePiece'+"""\ """
        input_data = np.array([(letter[1], f"{letter[0]-1/len(selfies_sentencepiece):.3f}") for letter in selfies_sentencepiece])
        input_data2 = np.array([(letter[1], f"{letter[0]:.3f}") for letter in selfies_sentencepiece])
        df = pd.concat([pd.DataFrame(data = input_data[:,1], index = input_data[:,0]).transpose(),pd.DataFrame(data = input_data2[:,1], index = input_data2[:,0]).transpose()]) 
        md += df.to_markdown()+"""\ """
    return md

def gather_attention_model(input_mols, tokenizer_suffix, model_path): # PREDICTION_MODEL_PATH/model_suffix/"checkpoint_last.pt"
    fairseq_dict_path = TASK_PATH / "bbbp" /tokenizer_suffix
    tokenizer = get_tokenizer(TOKENIZER_PATH /tokenizer_suffix)
    model = load_model(model_path, fairseq_dict_path,None)
    source_dictionary = get_dictionary(FAIRSEQ_PREPROCESS_PATH/tokenizer_suffix/"dict.txt")
    #preprocessed = model.encode(input_mols)
    attended = compute_attention_output(model, [input_mols], source_dictionary, tokenizer)
    return attended

def gather_attention(SMILES, smiles_atom_path=None, smiles_sentencepiece_path=None, selfies_atom_path=None, selfies_sentencepiece_path=None):
    SMILES = canonize_smile(SMILES)
    SELFIES = translate_selfie(SMILES)[0]
    if smiles_atom_path:
        smiles_atom = gather_attention_model(SMILES, "smiles_atom_isomers", smiles_atom_path)
    else:
        smiles_atom = [None]
    if smiles_sentencepiece_path:
        smiles_sentencepiece = gather_attention_model(SMILES, "smiles_trained_isomers", smiles_sentencepiece_path)
    else:
        smiles_sentencepiece = [None]
    if selfies_atom_path:
        selfies_atom = gather_attention_model(SELFIES, "selfies_atom_isomers", selfies_atom_path)
    else:
        selfies_atom = [None]
    if selfies_sentencepiece_path:
        selfies_sentencepiece = gather_attention_model(SELFIES, "selfies_trained_isomers", selfies_sentencepiece_path)
    else:
        selfies_sentencepiece = [None]
    mkd = [(1/len(SMILES), letter) for letter in SMILES]
    markdown = to_markdown(mkd,smiles_atom[0], smiles_sentencepiece[0], selfies_atom[0], selfies_sentencepiece[0])
    return markdown