from constants import TASK_PATH
from pathlib import Path
import pandas as pd
from embedding_classification import eval_weak_regressors, eval_weak_classifiers
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from ast import literal_eval
from utils import unpickle
from constants import ANNOT_PATH

def prepare_atom_annot():
    dataframe_smiles_bart = pd.read_csv(ANNOT_PATH/"embeds_smiles_BART.csv").set_index("SMILES") # first version "/scratch/ifender/SOS_tmp/embeddings_pretrainingdata/embeds_smiles_pretrained_BART_26_8.csv
    dataframe_smiles_roberta = pd.read_csv(ANNOT_PATH/"embeds_smiles_roberta.csv").set_index("SMILES") # first version "/scratch/ifender/SOS_tmp/embeddings_pretrainingdata/embeds_smiles_pretrained_roberta_26_8.csv
    dataframe_selfies_bart = pd.read_csv(ANNOT_PATH/"embeds_selfies_BART.csv").set_index("SELFIES") # first version "/scratch/ifender/SOS_tmp/embeddings_pretrainingdata/embeds_selfies_pretrained_BART_26_8.csv
    dataframe_selfies_roberta = pd.read_csv(ANNOT_PATH/"embeds_selfies_roberta.csv").set_index("SELFIES") # first version "/scratch/ifender/SOS_tmp/embeddings_pretrainingdata/embeds_selfies_pretrained_roberta_26_8.csv
    mapping = unpickle(Path(ANNOT_PATH/"smilestoatomtypestoselfies_dikt.pkl")) # first version "/scratch/ifender/SOS_tmp/embeddings_pretrainingdata/smilestoatomtypestoselfies_dikt_22_8.pkl"
    rows = []
    for smiles, smiles_info in mapping.items():
        smiles_bart_embeddings = literal_eval(dataframe_smiles_bart.loc[smiles]["embedding"])
        smiles_roberta_embeddings = literal_eval(dataframe_smiles_roberta.loc[smiles]["embedding"])
        selfies = smiles_info["selfies"]
        selfies_bart_embeddings = literal_eval(dataframe_selfies_bart.loc[selfies]["embedding"])
        selfies_roberta_embeddings = literal_eval(dataframe_selfies_roberta.loc[selfies]["embedding"])
        inverted_atom_dict = {val: key[0] for key, val in smiles_info["selfies_map"].items()}
        for pos_it, smiles_position_to_keep in enumerate(smiles_info['posToKeep']):
            curr_row = {"SMILES":smiles, "SMILES_pos": smiles_position_to_keep,"smiles_token":smiles_info["smi_clean"][pos_it] ,"label":smiles_info["atom_types"][pos_it]}
            smiles_bart_emb = {f"SMILES_BART_emb_{it}": emb for it, emb in enumerate(smiles_bart_embeddings[smiles_position_to_keep][0])}
            smiles_roberta_emb = {f"SMILES_roberta_emb_{it}": emb for it, emb in enumerate(smiles_roberta_embeddings[smiles_position_to_keep][0])}
            selfies_position_to_keep = inverted_atom_dict[smiles_position_to_keep]
            selfies_bart_emb = {f"SELFIES_BART_emb_{it}": emb for it, emb in enumerate(selfies_bart_embeddings[selfies_position_to_keep][0])}
            selfies_roberta_emb = {f"SELFIES_roberta_emb_{it}": emb for it, emb in enumerate(selfies_roberta_embeddings[selfies_position_to_keep][0])}
            curr_row = curr_row | smiles_bart_emb | smiles_roberta_emb | selfies_bart_emb | selfies_roberta_emb
            rows.append(curr_row)
    comb_dataframe = pd.DataFrame(rows)
    comb_dataframe["Element"] = comb_dataframe["smiles_token"].str.lstrip("[").str.upper().str.slice(0,1)
    comb_dataframe.to_csv(ANNOT_PATH/"comb_annotations.csv") # first version without ANNOT_PATH


def eval_eth():
    descriptors = ["mulliken", "resp1", "resp2", "dual", "mbis_dipole_strength"]    
    # all regression
    dataset = pd.read_csv("merged_eth_dataset.csv")
    groups = dataset["SMILES"].values
    for comb in ["SMILES_roberta", "SELFIES_roberta", "SMILES_BART", "SELFIES_BART"]:
        train_X =  dataset[[col for col in dataset.columns if comb in col]]
        for descriptor in descriptors:
            label = dataset[descriptor]
            eval_weak_regressors(train_X=train_X,train_y=label,test_X=None,test_y=None,report_prefix=TASK_PATH/descriptor/comb,val_X=None,val_y=None,groups=groups)

def eval_atom_annot():    
    # all classification
    if not (ANNOT_PATH/"comb_annotations.csv").exists():
        prepare_atom_annot()
    dataset = pd.read_csv(ANNOT_PATH/"comb_annotations.csv") # first version without ANNOT_PATH 
    c_groups = ["c3", "ca", "c", "cc", "cd", "c2"]
    n_groups = ["ns", "nb", "na"]
    o_groups = ["o", "os", "oh"]
    all_groups = c_groups+n_groups+o_groups
    for comb in ["SMILES_roberta", "SELFIES_roberta", "SMILES_BART", "SELFIES_BART"]:
        for curr_name, curr_group in [("annot_c", c_groups),("annot_n", n_groups),("annot_o", o_groups),("annot_all", all_groups), ]:
            curr_dataset = dataset[dataset["label"].isin(curr_group)]
            label = curr_dataset["label"]
            groups = curr_dataset["SMILES"]
            train_X = curr_dataset[[col for col in curr_dataset.columns if comb in col]]
            eval_weak_classifiers(train_X=train_X,train_y=label,test_X=None,test_y=None,report_prefix=TASK_PATH/curr_name/comb,val_X=None,val_y=None,groups=groups)

if __name__=="__main__":
    eval_atom_annot()
    eval_eth()
    