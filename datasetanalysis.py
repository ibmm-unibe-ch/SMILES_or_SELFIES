""" Dataset analysis
SMILES or SELFIES, 2022
"""
import logging
import pickle
import uuid
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import selfies
from rdkit import Chem
from tqdm import tqdm

from constants import CALCULATOR, DESCRIPTORS, PROCESSED_PATH, PROJECT_PATH

def check_dups(df):
    print("Checking for SMILES duplicates.. ",end="")
    boolean = df.duplicated(subset=['SMILES']).any()
    bool_series = df.duplicated(subset=['SMILES'])
    dup_numbers = df.duplicated(subset=['SMILES']).sum()
    if boolean==True:
        print("{} SMILES duplicates found.. ".format(dup_numbers),end="") #duplicates are correctly detected, has been tested
    else:
        print("No SMILES duplicates found.. ",end="")
    print("Returning cleaned dataframe")
    return df[~bool_series]                        #duplicates correctly removed from df, tested
        
def calc_average_lengths(df):
    print("Calculating average lengths of SMILES in characters.. ",end="")
    average_len_SMI=df["SMILES"].apply(len).mean()
    print(average_len_SMI)
    print("Calculating average lengths of SELFIES in characters.. ",end="")
    average_len_SEL=df["SELFIES"].apply(len).mean()
    print(average_len_SEL)
    print("Calculating average lengths of SELFIES in tokens.. ",end="")
    average_len_SEL_tok=df["SELFIES_length"].mean()
    print(average_len_SEL_tok)
    
    df["SMILES_length"]=df["SMILES"].str.len()
    print(df)
    return df
    
    

def read_file(input_file,desc):
    df=pd.read_csv(input_file,skiprows=1,names=desc) #skip first row when reading input
    #print(df)
    return df
            
    

if __name__ == "__main__":
    additional_descs=['SELFIES','SELFIES_length','SMILES']
    desc=DESCRIPTORS+additional_descs
    # print(desc)
    # print(len(desc))
    df=read_file("./test_dups.csv",desc)
    noSMIdups_df=check_dups(df)
    print(noSMIdups_df)
    df_withlengths=calc_average_lengths(noSMIdups_df)
    print(df_withlengths)
    