""" Dataset analysis
SMILES or SELFIES, 2022
"""
import logging
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from tqdm import tqdm
from collections import Counter
from constants import DESCRIPTORS, ANALYSIS_PATH


def create_desc_diagram(df,desc_list,name,outputpath): 
    avg_dict=dict()
    for item in desc_list:
        average_desc=df[item].mean()
        logging.info(f"Average {item} : {average_desc}")
        avg_dict[item]=average_desc
        
    fig, ax=plt.subplots()   
    if name=="fr":
        fig.set_size_inches(22,15) 
    x=list(avg_dict.keys())
    y=list(avg_dict.values())
    plt.yticks(np.arange(0,max(y)+1,2.0))
    if name=="fr":
        plt.yticks(np.arange(min(y),max(y)+1,0.1))
    ax.bar(range(len(avg_dict)), y, tick_label=x, color='black')
    ax.set_title('Diagram of average {} descriptors'.format(name))
    ax.set_xlabel('Descriptors')
    ax.set_ylabel('Frequency')
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.tight_layout()

    fig.savefig(outputpath/"AvgDescs_{}.pdf".format(name),bbox_inches="tight")

def create_length_hist(df, outputpath):
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
    fig=fig.savefig(outputpath/'hist_SELFIESvsSMILES_length.pdf')
    
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
    fig1=fig1.savefig(outputpath/'hist_SELFIESvsSMILES_length2.pdf')
    
    #histogram SELFIES tokens
    fig2, ax2=plt.subplots()
    df.hist(column='SELFIES_length_tok',ax=ax2, grid=False, color="black")
    ax2.set_title('Histogram of SELFIES length in tokens')
    ax2.set_xlabel('Length [No. of tokens]')
    ax2.set_ylabel('Frequency')
    fig2.savefig(outputpath/"SELFIES_tokenlength_hist.pdf")
    
    #histogram SELFIES characters
    fig3, ax3=plt.subplots()
    df.hist(column='SELFIES_length_char',ax=ax3, grid=False, color="black")
    ax3.set_title('Histogram of SELFIES length in characters')
    ax3.set_xlabel('Length [No. of characters]')
    ax3.set_ylabel('Frequency')
    fig3.savefig(outputpath/"SELFIES_charlength_hist.pdf")
    
    #histogram SMILES length
    fig4, ax4=plt.subplots()
    df.hist(column='SMILES_length',ax=ax4, grid=False, color="black")
    ax4.set_title('Histogram of SMILES length')
    ax4.set_xlabel('Length [No. of characters]')
    ax4.set_ylabel('Frequency')
    fig4.savefig(outputpath/"SMILES_length_hist.pdf")

def create_molweight_hist(df,outputpath):
    fig, ax=plt.subplots()
    df.hist(column='MolWt',ax=ax, grid=False, color="black")
    ax.set_title('Histogram of molecular weight')
    ax.set_xlabel('Molecular weight [g/mol]')
    ax.set_ylabel('Frequency')
    fig.savefig(outputpath/"MolWt_hist.pdf")
    
def create_hdonoracceptorhist(df,outputpath):
    fig, ax=plt.subplots()
    df.hist(column='NumHAcceptors',ax=ax, grid=False, color="black")
    ax.set_title('Histogram of number of H-Acceptors')
    ax.set_xlabel('Hydrogen acceptors in numbers')
    ax.set_ylabel('Frequency')
    fig.savefig(outputpath/"NumHAcceptors_hist.pdf")
    
    fig1, ax1=plt.subplots()
    df.hist(column='NumHDonors',ax=ax1, grid=False, color="black")
    ax1.set_title('Histogram of number of H-Donors')
    ax1.set_xlabel('Hydrogen donors in numbers')
    ax1.set_ylabel('Frequency')
    fig1.savefig(outputpath/"NumHDonors_hist.pdf")
    
    fig2, ax2=plt.subplots()
    df.hist(column='NumAromaticRings',ax=ax2, grid=False, color="black")
    ax2.set_title('Histogram of number of aromatic rings')
    ax2.set_xlabel('Aromatic rings in numbers')
    ax2.set_ylabel('Frequency')
    fig2.savefig(outputpath/"NumAromaticRings_hist.pdf")
    
def create_diagrams(df,outputpath):
   create_length_hist(df,outputpath)
   create_molweight_hist(df,outputpath)
   create_hdonoracceptorhist(df,outputpath)
   desc_list=['HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 
    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 
    'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 
    'RingCount', 'MolLogP']
   desc_fr_list=['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 
    'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 
    'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 
    'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 
    'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 
    'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 
    'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 
    'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 
    'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 
    'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
   create_desc_diagram(df,desc_list,"count",outputpath)
   create_desc_diagram(df,desc_fr_list,"fr",outputpath)
   

def check_dups(df):
    boolean = df.duplicated(subset=['SMILES']).any()
    bool_series = df.duplicated(subset=['SMILES'])
    dup_numbers = df.duplicated(subset=['SMILES']).sum()
    if boolean==True:
        logging.info(f"{dup_numbers} SMILES duplicates found..") #duplicates are correctly detected, has been tested
    else:
        logging.info("No SMILES duplicates found..")
    logging.info("Returning cleaned dataframe")
    return df[~bool_series]                        #duplicates correctly removed from df, tested
        
def calc_average_lengths(df):
   # print("Calculating average lengths of SMILES in characters.. ",end="")     
    average_len_SMI=df["SMILES"].apply(len).mean()
    logging.info(f"Calculating average lengths of SMILES in characters.. {average_len_SMI}")
    max_len_SMI=df["SMILES"].apply(len).max()
    logging.info(f"Calculating maximum length of SMILES in characters.. {max_len_SMI}")  
    average_len_SEL=df["SELFIES"].apply(len).mean()
    logging.info(f"Calculating average lengths of SELFIES in characters.. {average_len_SEL}")
    average_len_SEL_tok=df["SELFIES_length_tok"].mean()
    logging.info(f"Calculating average lengths of SELFIES in tokens.. {average_len_SEL_tok}")
    
    df.loc[:,"SMILES_length"]=df["SMILES"].str.len()
    df.loc[:,"SELFIES_length_char"]=df["SELFIES"].str.len()
    return df

def read_file(input_file,desc):
    df=pd.read_csv(input_file,skiprows=1,names=desc) #skip first row when reading input
    return df

if __name__ == "__main__":
    ANALYSIS_PATH.mkdir(parents=True, exist_ok=True)
    additional_descs=['SELFIES','SELFIES_length_tok','SMILES']
    desc=DESCRIPTORS+additional_descs
    df=read_file("./test.csv",desc)
    df_noSMIdups=check_dups(df)
    df_withlengths=calc_average_lengths(df_noSMIdups)
    create_diagrams(df_withlengths, ANALYSIS_PATH)
    
    