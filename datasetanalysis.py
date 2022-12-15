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
import matplotlib.pyplot as plt
import selfies
from rdkit import Chem
from tqdm import tqdm

from constants import CALCULATOR, DESCRIPTORS, PROCESSED_PATH, PROJECT_PATH

def create_length_hist(df):
    #histogram SELFIES tokens vs SMILES characters
    fig, ax = plt.subplots()
    a_heights, a_bins = np.histogram(df['SELFIES_length_tok'])
    b_heights, b_bins = np.histogram(df['SMILES_length'], bins=a_bins)
    width = (a_bins[1] - a_bins[0])/3
    #width=10
    ax.bar(a_bins[:-1], a_heights, width=width, facecolor='black', label="SELFIES") 
    ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='grey', label="SMILES")
    ax.set_title('Histogram SELFIES [in tokens] vs SMILES [in chars]')
    ax.set_xlabel('Length [No. of tokens or characters]')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig=fig.savefig('hist_SELFIESvsSMILES_length')
    
    #histogram SELFIES vs SMILES characters
    fig1, ax1 = plt.subplots()
    a1_heights, a1_bins = np.histogram(df['SELFIES_length_char'])
    b1_heights, b1_bins = np.histogram(df['SMILES_length'], bins=a1_bins)
    width = (a1_bins[1] - a1_bins[0])/3
    #width=10
    ax1.bar(a1_bins[:-1], a1_heights, width=width, facecolor='black', label="SELFIES") 
    ax1.bar(b1_bins[:-1]+width, b1_heights, width=width, facecolor='grey', label="SMILES")
    ax1.set_title('Histogram SELFIES vs. SMILES [in chars]')
    ax1.set_xlabel('Length [No. of characters]')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    fig1=fig1.savefig('hist_SELFIESvsSMILES_length2')
    
    #histogram SELFIES tokens
    fig2, ax2=plt.subplots()
    df.hist(column='SELFIES_length_tok',ax=ax2, grid=False, color="black")
    ax2.set_title('Histogram of SELFIES length in tokens')
    ax2.set_xlabel('Length [No. of tokens]')
    ax2.set_ylabel('Frequency')
    fig2.savefig("SELFIES_tokenlength_hist")
    
    #histogram SELFIES characters
    fig3, ax3=plt.subplots()
    df.hist(column='SELFIES_length_char',ax=ax3, grid=False, color="black")
    ax3.set_title('Histogram of SELFIES length in characters')
    ax3.set_xlabel('Length [No. of characters]')
    ax3.set_ylabel('Frequency')
    fig3.savefig("SELFIES_charlength_hist")
    
    #histogram SMILES length
    fig4, ax4=plt.subplots()
    df.hist(column='SMILES_length',ax=ax4, grid=False, color="black")
    ax4.set_title('Histogram of SMILES length')
    ax4.set_xlabel('Length [No. of characters]')
    ax4.set_ylabel('Frequency')
    fig4.savefig("SMILES_length_hist")

def create_molweight_hist(df):
    fig, ax=plt.subplots()
    df.hist(column='MolWt',ax=ax, grid=False, color="black")
    ax.set_title('Histogram of molecular weight')
    ax.set_xlabel('Molecular weight [g/mol]')
    ax.set_ylabel('Frequency')
    fig.savefig("MolWt_hist")
    
def create_hdonoracceptorhist(df):
    fig, ax=plt.subplots()
    df.hist(column='NumHAcceptors',ax=ax, grid=False, color="black")
    ax.set_title('Histogram of number of H-Acceptors')
    ax.set_xlabel('Hydrogen acceptors in numbers')
    ax.set_ylabel('Frequency')
    fig.savefig("NumHAcceptors_hist")
    
    fig1, ax1=plt.subplots()
    df.hist(column='NumHDonors',ax=ax1, grid=False, color="black")
    ax1.set_title('Histogram of number of H-Donors')
    ax1.set_xlabel('Hydrogen donors in numbers')
    ax1.set_ylabel('Frequency')
    fig1.savefig("NumHDonors_hist")
    
    fig2, ax2=plt.subplots()
    df.hist(column='NumAromaticRings',ax=ax2, grid=False, color="black")
    ax2.set_title('Histogram of number of aromatic rings')
    ax2.set_xlabel('Aromatic ringsin numbers')
    ax2.set_ylabel('Frequency')
    fig2.savefig("NumAromaticRings_hist")
    
def create_hists(df):
   create_length_hist(df)
   create_molweight_hist(df)
   create_hdonoracceptorhist(df)
   

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
    max_len_SMI=df["SMILES"].apply(len).max()
    print(average_len_SMI)
    print("Calculating maximum length of SMILES in characters.. ",end="")
    print(max_len_SMI)
    print("Calculating average lengths of SELFIES in characters.. ",end="")
    average_len_SEL=df["SELFIES"].apply(len).mean()
    print(average_len_SEL)
    print("Calculating average lengths of SELFIES in tokens.. ",end="")
    average_len_SEL_tok=df["SELFIES_length_tok"].mean()
    print(average_len_SEL_tok)
    
    df.loc[:,"SMILES_length"]=df["SMILES"].str.len()
    df.loc[:,"SELFIES_length_char"]=df["SELFIES"].str.len()
    #print(df)
    return df

def read_file(input_file,desc):
    df=pd.read_csv(input_file,skiprows=1,names=desc) #skip first row when reading input
    #print(df)
    return df

if __name__ == "__main__":
    additional_descs=['SELFIES','SELFIES_length_tok','SMILES']
    desc=DESCRIPTORS+additional_descs
    print(desc)
    # print(len(desc))
    df=read_file("./test.csv",desc)
    df_noSMIdups=check_dups(df)
    print(df_noSMIdups)
    df_withlengths=calc_average_lengths(df_noSMIdups)
    print(df_withlengths)
    create_hists(df_withlengths)
    