import os, sys
import json
import logging
from typing import List
from tokenisation import tokenize_dataset, get_tokenizer
from pathlib import Path
from fairseq_utils2 import compute_model_output, compute_model_output_RoBERTa, load_dataset, load_model
from fairseq.data import Dictionary
from SMILES_to_SELFIES_mapping import canonize_smiles, generate_mapping, generate_mappings_for_task_SMILES_to_SELFIES
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from constants import SEED
import numpy as np

from constants import (
    TASK_PATH,
    MOLNET_DIRECTORY,
    TOKENIZER_PATH
)

def build_legend(data):
    """
    Build a legend for matplotlib plt from dict
    """
    legend_elements = []
    for key in data:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=key,
                                      markerfacecolor=data[key], markersize=9))
    return legend_elements

def plot_lda(embeddings, labels, colours_dict, save_path, alpha=0.2):
    """Performing Linear Discriminant Analysis and plotting it

    Args:
        embeddings (_list[float]_): Embeddings of one element or a subgroup
        labels (_list[string]_): List of assigned atom types
        colours_dict (_dict[string][int]_): Dictionary of colors linking atomtypes to colors
        save_path (_string_): Path where to save plot
        alpha (float, optional): Level of opacity. Defaults to 0.2.
    """
    logging.info("Started plotting LDA")
    os.makedirs(save_path.parent, exist_ok=True)
    lda = LDA(n_components=2)
    lda_embeddings = lda.fit_transform(embeddings,labels)
    #logging.info(
    #    f"{save_path} has the explained variance of {pca.explained_variance_ratio_}"
    #)
    fig, ax = plt.subplots(1)
    ax.scatter(lda_embeddings[:, 0], lda_embeddings[:, 1], marker='.', alpha=alpha, c=[
               colours_dict[x] for x in labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(1.13, 0.5), fontsize=8)
    ax.set_ylabel("LDA 2", fontsize=17)
    ax.set_xlabel("LDA 1", fontsize=17)
    ax.set_title("LDA - Embeddings resp. atom types", fontsize=21)
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(length=8, width=3, labelsize=15)
    fig.savefig(f"{save_path}.svg", format="svg", bbox_inches='tight', transparent=True)
    fig.clf()
    
    # same but random labels
    lda = LDA(n_components=2)
    random_labels=labels.copy()
    np.random.shuffle(random_labels)
    lda_embeddings = lda.fit_transform(embeddings,random_labels)
    #logging.info(
    #    f"{save_path} has the explained variance of {pca.explained_variance_ratio_}"
    #)
    fig, ax = plt.subplots(1)
    ax.scatter(lda_embeddings[:, 0], lda_embeddings[:, 1], marker='.', alpha=alpha, c=[
               colours_dict[x] for x in random_labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(1.13, 0.5), fontsize=8)
    ax.set_ylabel("LDA 2", fontsize=17)
    ax.set_xlabel("LDA 1", fontsize=17)
    ax.set_title("LDA random - Embeddings resp. atom types", fontsize=21)
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(length=8, width=3, labelsize=15)
    fig.savefig(f"{save_path}_random.svg", format="svg", bbox_inches='tight', transparent=True)
    fig.clf()


def plot_pca(embeddings, labels, colours_dict, save_path, alpha=0.2):
    """Performing PCA and plotting it

    Args:
        embeddings (_list[float]_): Embeddings of one element or a subgroup
        labels (_list[string]_): List of assigned atom types
        colours_dict (_dict[string][int]_): Dictionary of colors linking atomtypes to colors
        save_path (_string_): Path where to save plot
        alpha (float, optional): Level of opacity. Defaults to 0.2.
    """
    logging.info("Started plotting PCA")
    os.makedirs(save_path.parent, exist_ok=True)
    pca = PCA(n_components=2, random_state=SEED + 6541)
    pca_embeddings = pca.fit_transform(embeddings)
    logging.info(
        f"{save_path} has the explained variance of {pca.explained_variance_ratio_}"
    )
    explained_variance_percentages = [f"{var:.2%}" for var in pca.explained_variance_ratio_]  # Format as percentages
    fig, ax = plt.subplots(1)
    ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], marker='.', alpha=alpha, c=[
               colours_dict[x] for x in labels])
    legend_elements = build_legend(colours_dict)
    ax.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(1.13, 0.5), fontsize=8)
    ax.set_ylabel(f"PCA 2, var {explained_variance_percentages[1]}", fontsize=17)
    ax.set_xlabel(f"PCA 1, var {explained_variance_percentages[0]}", fontsize=17)
    ax.set_title("PCA - Embeddings resp. atom types", fontsize=21)
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(length=8, width=3, labelsize=15)
    fig.savefig(f"{save_path}.svg", format="svg", bbox_inches='tight', transparent=True)
    fig.clf()

def plot_umap_pca_lda(p_f_cl_list_embs, p_f_cl_list_assigs, namestring, save_path_prefix, atomtype2color, min_dist, n_neighbors, alpha):
    #create paths on what to name the plots
    pathway = Path(str(save_path_prefix) +
                   f"umap_{min_dist}_{n_neighbors}_{namestring}")
    pathway_pca = Path(str(save_path_prefix) + f"pca_{namestring}")
    pathway_lda = Path(str(save_path_prefix) + f"lda_{namestring}")
    
 
    # plot PCA
    plot_pca(p_f_cl_list_embs, p_f_cl_list_assigs,
             atomtype2color[namestring], pathway_pca, alpha)
    # plot LDA
    #plot_lda(p_f_cl_list_embs, p_f_cl_list_assigs,
    #         atomtype2color[namestring], pathway_lda, alpha)
    # plot UMAP
   # plot_umap(p_f_cl_list_embs, p_f_cl_list_assigs, atomtype2color, pathway, min_dist, n_neighbors, alpha)

def plot_plots(atomtype_embedding_perelem_dict, colordict, min_dist, n_neighbors, alpha, save_path_prefix):
    namestring="c"
    plot_umap_pca_lda(atomtype_embedding_perelem_dict[namestring][1], atomtype_embedding_perelem_dict[namestring][0], namestring, save_path_prefix, colordict, min_dist, n_neighbors, alpha)
    
    namestring="p f cl o s"
    plot_umap_pca_lda(atomtype_embedding_perelem_dict[namestring][1], atomtype_embedding_perelem_dict[namestring][0], namestring, save_path_prefix, colordict, min_dist, n_neighbors, alpha)
    
    namestring = "c o"
    plot_umap_pca_lda(atomtype_embedding_perelem_dict[namestring][1], atomtype_embedding_perelem_dict[namestring][0], namestring, save_path_prefix, colordict, min_dist, n_neighbors, alpha)
    
    print("Plotting................................BY ELEMENT")
    # plot all atomtypes of one element only
    for key in colordict.keys():
        if len(key)<=2 and key!='cl' and key in atomtype_embedding_perelem_dict.keys():
            print(f"#######KEY {key}\n")
            pathway_umap = Path(str(save_path_prefix) +
                                f"umap_{min_dist}_{n_neighbors}_{key}.svg")
            pathway_pca = Path(str(save_path_prefix) + f"pca_{key}.svg")
            pathway_lda = Path(str(save_path_prefix) + f"lda_{key}.svg")
            embeddings = atomtype_embedding_perelem_dict[key][1]
            assignments = atomtype_embedding_perelem_dict[key][0]
            #atomtype2color, set_list = getcolorstoatomtype(set(assignments.copy()))

            try:
                assert len(embeddings) == (len(assignments)), "Assignments and embeddings do not have same length."
                assert len(embeddings)>10, "Not enough embeddings for plotting"
                print(f"len embeddings of key {key}: {len(embeddings)}")
                plot_umap_pca_lda(embeddings, assignments, key, save_path_prefix, colordict, min_dist, n_neighbors, alpha)
                #plot_pca(embeddings, assignments, atomtype2color, pathway_pca, alpha)
                #plot_lda(embeddings, assignments, atomtype2color, pathway_lda, alpha)
                #plot_umap(embeddings, assignments, atomtype2color, pathway_umap, min_dist, n_neighbors, alpha)
            except AssertionError as e:
                print(f"Assertion error occurred for element {key}: {e}")
                continue 

def get_atomtype_embedding_perelem_dict(filtered_dict, colordict, entry_name_atomtype_to_embedding):
    atomtype_to_embedding_lists = [value[entry_name_atomtype_to_embedding] for value in filtered_dict.values() if entry_name_atomtype_to_embedding in value and value[entry_name_atomtype_to_embedding] is not None]
    print("len atomtype to embedding list smiles: ",len(atomtype_to_embedding_lists))
    
    # sort embeddings according to atomtype, I checked it visually and the mapping works
    embeddings_by_atomtype = {}  # Dictionary to hold lists of embeddings for each atom type
    #listembeddings = list()
    for atom_type_list in atomtype_to_embedding_lists:
        # go through single dictionary
        for tuple in atom_type_list:
           # print(f"atomtype {atom_type} embeddings {embeddings[1]}")
            if tuple[0] not in embeddings_by_atomtype:
                embeddings_by_atomtype[tuple[0]] = []
            # extend the list of embeddings for this atom type(, but ONLY by the embedding not the attached token)
            embeddings_by_atomtype[tuple[0]].append(tuple[1][0])
            #print("\ntuple01",len(tuple[1][0]),tuple[1][0])
            #print(len(embeddings[0]))
    print("embeddings c",len(embeddings_by_atomtype['c']))
    
    # sort dictionary that is mapping embeddings to atomtypes to elements so that e.g. all carbon atom types can be accessed at once in one list
    #atom_types_repeated = []
    #embeddings_list = []
    atomtype_embedding_perelem_dict = dict()
    ctr = 0
    for key in colordict.keys():
        print(f"key {key}")
        for atype in colordict[key]:
            print(atype) 
            if atype in embeddings_by_atomtype.keys():
                embsofatype = embeddings_by_atomtype[atype]
                atypes = [atype] * len(embeddings_by_atomtype[atype])
                assert len(embsofatype) == len(atypes), "Length of embeddings and atom types do not match."
                if key not in atomtype_embedding_perelem_dict:
                    atomtype_embedding_perelem_dict[key] = ([],[])
                if key in atomtype_embedding_perelem_dict:
                    atomtype_embedding_perelem_dict[key][0].extend(atypes)
                    atomtype_embedding_perelem_dict[key][1].extend(embsofatype)
    
    print(atomtype_embedding_perelem_dict.keys())
    return atomtype_embedding_perelem_dict

def create_plotsperelem(dikt, colordict, penalty_threshold, min_dist, n_neighbors, alpha, save_path_prefix):
    """Create plot per element and for all element subsets

    Args:
        dikt (_dict_): Dictionary of atom mappings etc
        colordict (_dict[string][dict[string],[color]]): Dictionary that maps atom types to colors
        penalty_threshold (_float_): Threshold for max penalty score
        min_dist (_float_): Number of min dist to use in UMAP
        n_neighbors (_int_): Number of neighbors to use in UMAP
        alpha (_int_): Level of opacity
        save_path_prefix (_string_): Path prefix where to save output plot
    """
    print(colordict.keys())
    cs=0
    #for key,val in dikt.items():
    #    if dikt[key]['atomtype_to_embedding'] is not None:
    #        #dikt[key]['atomtype_to_embedding']['c']
            #print(dikt[key]['atomtype_to_embedding'])
    #        for tuple in dikt[key]['atomtype_to_embedding']:
    #            print("%%%",tuple[0],len(tuple[1][0]), tuple[1][1])
    #            print()
    #            print()
    #        print(dikt[key]['atom_types'])
    #        cs+=1
    print("cs-----------------",cs)   
    # Assuming 'dikt' is your dictionary and each value has a 'penalty_score' key
    filtered_dict_filterthresh = {smiles: info for smiles, info in dikt.items() if info['max_penalty'] is not None and info['max_penalty'] < penalty_threshold}
    
    # there are less SELFIES embeddings that could be mapped to the SMILES and thereby the atomtypes, so the number of embeddings is less
    # therefore filter filtered_dict further on what is available for SELFIES and also what is available for SMILES
    filtered_dict = {smiles: info for smiles, info in filtered_dict_filterthresh.items() if info['atomtype_to_embedding'] is not None and info['atomtype_to_clean_selfies_embedding'] is not None}
    print("keys in filtered dict:",len(filtered_dict.keys()))
    #for key, value in filtered_dict.items():
    #    print(value['max_penalty'])
    
    # -------------------------SMILES
    atomtype_embedding_perelem_dict_smiles = get_atomtype_embedding_perelem_dict(filtered_dict, colordict, 'atomtype_to_embedding')
    print(f"len of atomtype embs per elem smiles: {len(atomtype_embedding_perelem_dict_smiles)}")
    plot_plots(atomtype_embedding_perelem_dict_smiles, colordict, min_dist, n_neighbors, alpha, f"{save_path_prefix}smiles")
    
    #------------------------------SELFIES 
    print("plotting SELFIES")
    atomtype_embedding_perelem_dict_selfies = get_atomtype_embedding_perelem_dict(filtered_dict, colordict, 'atomtype_to_clean_selfies_embedding')
    print(f"len of atomtype embs per elem selfies: {len(atomtype_embedding_perelem_dict_selfies)}")
    plot_plots(atomtype_embedding_perelem_dict_selfies, colordict, min_dist, n_neighbors, alpha, f"{save_path_prefix}selfies")

def colorstoatomtypesbyelement(atomtoelems_dict):
    """Generating a dictionary of colors given a dictionary that maps atomtypes to elements

    Args:
        atomtoelems_dict (_dict_): Dictionary that maps atom types to elements

    Returns:
        _dict,: Dictionary that maps atom types to colors
    """
    # https://sashamaps.net/docs/resources/20-colors/ #95% accessible only, subject to change, no white
    colors_sash = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
                   '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000']
   
    colordict = dict()
    for key in atomtoelems_dict.keys():
        atypes = atomtoelems_dict[key]
        keycoldict=dict()
        for at, col in zip(atypes, colors_sash[0:len(atypes)]):
            keycoldict[at]=col    
        colordict[key]=keycoldict 
    print(colordict.items())
    
    # now instead for each element, get colors for a combination of atomtypes
    # p f cl o s
    key='p f cl o s'
    pfclos_types = atomtoelems_dict['p']+atomtoelems_dict['f']+atomtoelems_dict['cl']+atomtoelems_dict['o']+atomtoelems_dict['s']
    keycoldicti=dict()
    for at, col in zip(pfclos_types, colors_sash[0:len(pfclos_types)]):
        keycoldicti[at]=col
    colordict[key]=keycoldicti 
    # c o
    key='c o'
    pfclos_types = atomtoelems_dict['c']+atomtoelems_dict['o']
    keycoldicti=dict()
    for at, col in zip(pfclos_types, colors_sash[0:len(pfclos_types)]):
        keycoldicti[at]=col
    colordict[key]=keycoldicti 
    print(colordict.keys())
    print(colordict.items())
    return colordict
    

def create_elementsubsets(atomtype_set):
    """Creation of element subsets according to alphabet

    Args:
        big_set (_set_): Set of atom types
    Returns:
        _list,dict[string][list[float],list[string]]_: List of keys (elements), dictionary that contains atomtypes sorted by element
    """
    atomtype_set=sorted(atomtype_set)
    element_dict = dict()
    elements = list()
    ctr=0
    last_firstval = ''
    for atype in atomtype_set:
        if ctr==0:
            last_firstval = atype[0]
        if not atype.startswith('cl') and atype not in element_dict.items() and atype[0]==last_firstval:
            #print(elements)
            elements.append(atype)
            element_dict[last_firstval] = elements
        elif last_firstval != atype[0] and atype != 'cl' and atype != 'br':
            element_dict[last_firstval] = elements
            elements = list()
            elements.append(atype)
            last_firstval = atype[0]
        ctr+=1
    element_dict['cl']=['cl']
    element_dict['br']=['br']
    return element_dict

def map_selfies_embeddings_to_smiles(embeds_selfies, smiles_to_selfies_mapping, dikt):
    """Map  clean SELFIES embeddings to their corresponding SMILES and atomtypes
    Args:
        embeds_selfies (_list_): List of lists of SELFIES embeddings
        smiles_to_selfies_mapping (_dict_): Dictionary that maps SMILES to SELFIES and SELFIES tokens to SMILES tokens (mappings[smiles]['selfiesstr_tok_map'] = (selfies_str,tokenised_selfies,mapping))
        dikt (_dict_): Dictionary of atom mappings etc
    Returns:
        adds SELFIES embeddings to atomtype mappings to dikt
    """
    # get embeddings for SELFIES that have a mapping to SMILES and map to SMILES in smiles_to_selfies_mapping
    for emb, smiles in zip(embeds_selfies[0], smiles_to_selfies_mapping.keys()):
        # Check if the mapping for the current smiles has a non-None value at index 2 for mapping of SELFIES to SMILES
        if smiles_to_selfies_mapping[smiles]['selfiesstr_tok_map'][2] is not None:
            # If so, set 'selfies_emb' to emb, otherwise set it to None
            smiles_to_selfies_mapping[smiles].setdefault("raw_selfies_emb", emb)
        else:
            smiles_to_selfies_mapping[smiles].setdefault("raw_selfies_emb", None)

    print("within", len(dikt.keys()))
    for key,val in dikt.items():
        #print("smiles:",key, val['atomtype_to_embedding'][0])
        if key in smiles_to_selfies_mapping.keys():
            # get list with positions to keep from dikt
            #if assignment failed posToKeep will be empty, then there is no need to map anything
            posToKeep = dikt[key]["posToKeep"]
            if posToKeep is not None:
               # print("selfies:", smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][0], smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][1])
                # if mapping exists
                if smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][2] is not None and smiles_to_selfies_mapping[key]['raw_selfies_emb'] is not None:
                   # print("key:",key)
                   # print("1111111111: ",smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][1])
                   # print("1111111111: ",smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][2].keys())
                    # 1 atom mappings and number of embeddings do not have to be even because branch, ring and overloaded tokens cannot be mapped to tokens in canonized SMILES
                    # 1 keep only the mebeddings that have a mapping
                    embs_with_mapping = []
                    for x, val in smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][2].items():
                        token_id = x[0]
                        #print("\ttoken:",x[1])
                        #print("\ttoken id:",x[0]) 
                        #print("\t in embedding: ",smiles_to_selfies_mapping[key]['raw_selfies_emb'][token_id][1])
                        #print()
                        #print("maps to smiles id: ",val)
                        #print()
                        assert x[1]==smiles_to_selfies_mapping[key]['raw_selfies_emb'][token_id][1], f"Token {x[1]} does not match token {smiles_to_selfies_mapping[key]['raw_selfies_emb'][token_id][1]}"
                        embs_with_mapping.append((val, smiles_to_selfies_mapping[key]['raw_selfies_emb'][token_id]))
                        #embs_with_mapping.append(smiles_to_selfies_mapping[key]['raw_selfies_emb'][token_id])
                    # 2 resort embeddings according to their position in the SMILES string
                    embs_with_mapping = sorted(embs_with_mapping, key=lambda item: item[0])
                    #print("sorted ", [key for key, _ in embs_with_mapping])
                   ## for _, value in embs_with_mapping:
                     #   print(value[1])
                    # 3 only keep embeddings with smiles id that belong to id that is in posToKeepList
                    filtered_embs = [(key, value) for key, value in embs_with_mapping if key in posToKeep]
                    # 4 assert that the length of the filtered embeddings is the same as the length of the posToKeep list
                    assert len(filtered_embs) == len(posToKeep), f"Length of filtered embeddings {len(filtered_embs)} and posToKeep list {len(posToKeep)} do not agree."
                    # 5 map the filtered embeddings to the atom types
                    atomtypes= dikt[key]['atom_types']
                    assert len(atomtypes) == len(filtered_embs), f"Length of atom types {len(atomtypes)} and filtered embeddings {len(filtered_embs)} do not agree."
                    atomtypes_to_selfies_embs = []
                    for atomtype, emb in zip(atomtypes, filtered_embs):
                        atomtypes_to_selfies_embs.append((atomtype, emb[1]))  
                        # assert letters of atomtype and token of embedding match
                        # checked visually, looks good
                        #print("----------------------------------------------------")
                        #print(f"atomtype {atomtype} emb {emb[1][1]}")
                        
                    # 6 attach this dictionary with name 'atomtype_to_clean_selfies_embedding' to the dikt
                    dikt[key].setdefault("atomtype_to_clean_selfies_embedding", atomtypes_to_selfies_embs) 
                else:
                    dikt[key].setdefault("atomtype_to_clean_selfies_embedding", None)
            else:
                dikt[key].setdefault("atomtype_to_clean_selfies_embedding", None)
        else:
            dikt[key].setdefault("atomtype_to_clean_selfies_embedding", None)    

def map_embeddings_to_atomtypes(dikt,task_SMILES):
    for SMILES in task_SMILES:
        if dikt[SMILES]["posToKeep"] is not None:
            atomtype_to_embedding = {}
            atom_types = dikt[SMILES]['atom_types']
            embeddings = dikt[SMILES]['clean_embedding']
            type_to_emb_tuples_list = list()
            for atom_type, embedding in zip(atom_types, embeddings):
                type_to_emb_tuples_list.append((atom_type, embedding))
                #atomtype_to_embedding.setdefault(atom_type, []).append(embedding)
                #type_to_emb_dict[atom_type] = embedding
                # assert to check whether atom type is the same as the first letter of the embedding
                assert(atom_type.lower() if atom_type.lower() =='cl' or atom_type.lower() =='br' else atom_type[0].lower()==(embedding[1][1].lower() if embedding[1].startswith("[") else embedding[1]).lower()), f"Atom assignment failed: {atom_type} != {embedding[1]}"
            dikt[SMILES]["atomtype_to_embedding"] = type_to_emb_tuples_list
        else:
            dikt[SMILES]["atomtype_to_embedding"]= (None,None)
    logging.info("Embeddings mapped to atom types, all checks passed")

def get_clean_embeds(embeds, dikt, creation_assignment_fails, task_SMILES):
    """Clean embeddings of embeddings that encode for digits, hydrogens, or structural tokens

    Args:
        embeds (_List[List[float]_): Embeddings of a SMILES
        failedSmiPos (_list_): Positions of SMILES in list where no file and/or assignment could be generated
        posToKeep_list (_list_): List of positions in a SMILES according to tokens that need to be kept (not digits, hydrogens, or structural tokens)

    Returns:
        _list[float]_: Embeddings that do not encode hydrogens, digits, or structural tokens, but only atoms
    """
    # some sanity checks on embeddings per SMILES
    assert (len(dikt.keys())) == (len(
        embeds[0])), f"Number of SMILES and embeddings do not agree. Number of SMILES: {len(dikt.keys())} of which {creation_assignment_fails} failures and Number of embeddings: {len(embeds[0])}"
    print(f"Number of SMILES: {len(dikt.keys())} with {creation_assignment_fails} failures and Number of embeddings: {len(embeds[0])}")
    
    none_embeddings = sum([1 for emb in embeds[0] if emb is None])
    print("Sum of NONE embeddings:",none_embeddings)
    
    none_sth =0
    #only keep embeddings for SMILES where atoms could be assigned to types
    embeds_clean = list()
    for smi, emb in zip(task_SMILES, embeds[0]):
        posToKeep = dikt[smi]["posToKeep"]
        # new: embeddings can be none too
        if posToKeep is not None and emb is not None:
            embeds_clean.append(emb)
            dikt[smi]["orig_embedding"]=emb
        else:
            dikt[smi]["orig_embedding"]=None
            none_sth+=1
    
    logging.info(
        f"Length embeddings before removal: {len(embeds[0])}, after removal where atom assignment failed or embedding is None: {len(embeds_clean)}")
    creation_assignment_fails_AND_none_embeddings =creation_assignment_fails+none_embeddings
    numberdel_embs = (len(embeds[0])-len(embeds_clean))
    assert none_sth == (len(
        embeds[0])-len(embeds_clean)), f"Assignment fails ({creation_assignment_fails}) plus none embeddings {none_embeddings} (leads to: {none_sth}) and number of deleted embeddings do not agree ({numberdel_embs})."

    embeds_cleaner = []
    #assert len(embeds_clean) == (len([item for item in posToKeep_list if item is not None])
     #                            ), f"Not the same amount of embeddings as assigned SMILES. {len(embeds_clean)} embeddings vs. {len([item for item in posToKeep_list if item is not None])} SMILES with positions"
    # only keep embeddings that belong to atoms
    for SMILES in task_SMILES:
        poslist = dikt[SMILES]["posToKeep"]
        emb_clean = dikt[SMILES]["orig_embedding"]

        if poslist is not None and emb_clean is not None:
            newembsforsmi = []
            newembsforsmi = [emb_clean[pos] for pos in poslist]
            embeds_cleaner.append(newembsforsmi)
            dikt[SMILES]["clean_embedding"]=newembsforsmi  
        else:
            # if original embeddings is None, make clean embedding None too
            dikt[SMILES]["clean_embedding"]=None   
            # also set posToKeep to None
            dikt[SMILES]["posToKeep"]=None

    # sanity check that length of embeddings to keep is the same as length of embeddings to keep
    posToKeep_list = [value["posToKeep"] for value in dikt.values() if value["posToKeep"] is not None]
    # sanity check that the lengths agree
    for smiemb, pos_list in zip(embeds_cleaner, posToKeep_list):
        assert len(smiemb) == len(
            pos_list), "Final selected embeddings for assigned atoms do not have same length as list of assigned atoms."
        #print(len(smiemb), pos_list)
        
    # sanity check that length of assigned atoms map to length of clean embeddings
    for SMILES in task_SMILES:
        smi_clean=dikt[SMILES]["smi_clean"]
        emb_clean = dikt[SMILES]["clean_embedding"]
        if dikt[SMILES]["posToKeep"] is not None and emb_clean is not None:
            assert len(smi_clean) == len(
                emb_clean), "SMILES and embeddings do not have same length."
            for sm, em in zip(smi_clean,emb_clean):
                #print(f"sm {sm} em {em[1]}")
                assert(sm==em[1]), f"Atom assignment failed: {sm} != {em[1]}"
    logging.info("Cleaning embeddings finished, all checks passed")
    return embeds_cleaner

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

    #data_path = "/data/jgut/SMILES_or_SELFIES/task/delaney/smiles_atom_isomers"
    model = load_model(specific_model_path, data_path, cuda)
    #print("model loaded")
    model.zero_grad()
    data_path = data_path / "input0" / "test"
    # True for classification, false for regression
    dataset = load_dataset(data_path, True)
    source_dictionary = Dictionary.load(str(data_path.parent / "dict.txt"))

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
        
    if traintype=="finetuned":
        if model=="BART":
            # path for BART  
            specific_model_path = (
            finetuned_TASK_MODEL_PATH
            / task
            / f"{subfolder}_bart"
            / "1e-05_0.2_seed_0" 
            / "checkpoint_best.pt"
            )
        else:
            if rep=='selfies':
                #path for RoBERTa
                specific_model_path = (
                    finetuned_TASK_MODEL_PATH
                    / task
                    / f"{subfolder}_roberta"
                    / "5e-06_0.2_seed_0" 
                    / "checkpoint_best.pt"
                )
            else:
                #path for RoBERTa
                specific_model_path = (
                    finetuned_TASK_MODEL_PATH
                    / task
                    / f"{subfolder}_roberta"
                    / "1e-05_0.2_seed_0" 
                    / "checkpoint_best.pt"
                )
    # ----------------------specific model paths for pretrained models of BART and RoBERTa-------------------------
    elif traintype=="pretrained":
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
    data_path = TASK_PATH / task / f"{subfolder}"
    
    embeds = []
    embeds = get_embeddings(task, specific_model_path, data_path, False, reps) #works for BART model with newest version of fairseq on github, see fairseq_git.yaml file
    checked_embeds = check_lengths(listoftokenisedreps, embeds) #, "Length of SMILES_tokens and embeddings do not agree."
    print("got the embeddings")
    return checked_embeds


def get_tokenized_SMILES(task_SMILES: List[str]):
    """Tokenize SMILES string

    Args:
        input_list of strings (str): List of SMILES input string

    Returns:
        dict: dictionary that links canonize SMILES string
    """

    tokenizer = get_tokenizer(TOKENIZER_PATH)
    print(f"tokenizer {tokenizer}")
    smi_toks = tokenize_dataset(tokenizer, task_SMILES, False)
    smi_toks = [smi_tok.split() for smi_tok in smi_toks]
    print(f"SMILES tokens: {smi_toks[0]}")
    smiles_dict = dict(zip(task_SMILES,smi_toks))
    return smiles_dict

def load_dictsandinfo_from_jsonfolder(input_folder):
    """
    Load atom assignments and info on failed assignments from folder that contains dictionaries on antechamber atom assignments and info files on failed assignments
    :param input_folder: folder that contains atom assignments and info on failed assignments
    :return: dict of tasks with dictionary with atom assignments, total number of failed assignments, list of failed SMILES and positions that failed, list of positions that should be kept
    """
    task_dikt = {}
    task_totalfails = {}
    for file in os.listdir(input_folder):
        if file.endswith(".json"):
            if file.startswith("dikt"):
                task = file.split("_")[1].split(".")[0]
                if task=="bace":
                    task = "bace classification"
                with open(os.path.join(input_folder, file), 'r') as f:
                    data = json.load(f)
                    task_dikt[task] = data
                    #totalfails += data['totalfails']
                    #failedSmiPos.extend(data['failedSmiPos'])
                    #posToKeep_list.extend(data['posToKeep'])
            elif file.startswith("assignment_info"):
                task=file.split(".")[0].split("_")[2]
                if task=="bace":
                    task = "bace classification"
                with open(os.path.join(input_folder, file), 'r') as f:
                    data = json.load(f)
                    task_totalfails[task] = data
                    #failedSmiPos.extend(data['failedSmiPos'])
                    #posToKeep_list.extend(data['posToKeep'])
                
    return task_dikt, task_totalfails


if __name__ == "__main__":

    # get atom assignments from folder that contains antechamber atom assignments and info files on failed assignments
    input_folder = "./assignment_dicts"
    task_dikt, task_totalfails = load_dictsandinfo_from_jsonfolder(input_folder)
    #currently available processed tasks 
    #task = "delaney" --> regression
    #task = "bace_classification" --> only fails
    #task="bbbp" --> classification
    #task="clearance" --> regression
    onlyworkingtasks=["delaney", "clearance", "bbbp","lipo"]
    test_tasks=["delaney"]
    

    merged_dikt = {}
    #print(f"totalfails: {totalfails} and total assigned molecules: {len(dikt.keys())}")
    for key, val in task_dikt.items():
        print("TASK: ",key)
        task=key
        if task not in onlyworkingtasks:
            continue
        #print(f"SMILES task: {key} \nwith dict_keys {val.keys()}")
        #print(task_dikt[key].keys())
        dikt=task_dikt[key]
        totalfails = task_totalfails[key]['totalfails']
        task_SMILES=dikt.keys()
        
        # have to tokenize SMILES just like previously in e.g. 2_AssignEmbedsPlot.py
        smiles_dict = get_tokenized_SMILES(task_SMILES)
        #for key2, val2 in val.items():
        #    print(f"{key2}: {val2}")
        percentagefailures = (totalfails/len(dikt.keys()))*100
        print(f"total fails for task {task}: {totalfails} out of {len(dikt.keys())} SMILES ({percentagefailures:.2f}%) ")
        #get embeddings from model
        #model = "ROBERTA"
        model="BART"
        traintype = "pretrained"
        rep = "smiles"
        # task needs specifiyng for loading of finetuned model
        task = key
        ########################## Get embeddings from model for SMILES ############################################
        try:
            print("get embeddings")
            embeds = get_embeddings_from_model(task, traintype, model, rep, smiles_dict.keys(), smiles_dict.values())
            #get rid of embeddings that encode for digits or hydrogens
            embeds_clean = get_clean_embeds(embeds, dikt, totalfails, task_SMILES)
            # within the dikt, map embeddings to atom types
            map_embeddings_to_atomtypes(dikt,task_SMILES)
            print()
            
            ## SELFIES------------------------------------------------------------------------------------------------------------------------------------------------------------

            # get SELFIES equivalent of SMILES and mapping between them
            smiles_to_selfies_mapping = generate_mappings_for_task_SMILES_to_SELFIES(task_SMILES)
            
            selfies_tokenised = []
            selfies = []
            maps_num = 0
            for key in smiles_to_selfies_mapping.keys():
                print(f"SMILES: {key} SELFIES: {smiles_to_selfies_mapping[key]}")
                selfies_tokenised.append(smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][1])
                selfies.append(smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][0])
                if smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][2] is not None:
                    maps_num +=1
                    for key2,val in smiles_to_selfies_mapping[key]['selfiesstr_tok_map'][2].items():
                        print(f"SELFIES index:{key2[0]} with token:{key2[1]}\tmaps to SMILES token at pos: {val}")
                print()
                
            print(f"list of tokenised selfies: {selfies_tokenised}")
            print(f"selfies {selfies} \nwith len() {len(selfies)}")
            print(f"mappings {maps_num}")
            
            rep="selfies"
            # traintype and model speicfied above
            embeds_selfies = get_embeddings_from_model(task, traintype, model, rep, selfies, selfies_tokenised)
            
            # map selfies embeddings to smiles in smiles_dict
            map_selfies_embeddings_to_smiles(embeds_selfies, smiles_to_selfies_mapping, dikt)
            for key, val in dikt.items():
                dikt[key]['task']=task
            
            
        except Exception as e:
            print(f"Error: {e}")
            continue
        merged_dikt.update(dikt)

    
    print(len(merged_dikt.keys()))
    print(merged_dikt.keys())
    valid_keys_count = len([key for key in merged_dikt.keys() if merged_dikt[key]['posToKeep'] is not None and merged_dikt[key]['atomtype_to_embedding'] is not None and merged_dikt[key]['atomtype_to_clean_selfies_embedding'] is not None])
    print("==============================================================================================================================================")
    print(f"Number of valid keys in final merged_dikt: {valid_keys_count}")

    # following this, the dict looks as follows: atomtype_to_dict should be a list of tuples with atomtype and embeddings    
    # dikt[SMILES] with dict_keys(['posToKeep', 'smi_clean', 'atom_types', 'max_penalty', 'orig_embedding', 'clean_embedding', 'atomtype_to_embedding', 'atomtype_to_clean_selfies_embedding'])

    # SELFIES embeddings mapped to atomtypes-------------------------------------------------------------------------------------------------------------------------     
    # following this, the dict looks as follows: atomtype_to_dict should be a list of tuples with atomtype and embeddings    
    # dikt[SMILES] with dict_keys(['posToKeep', 'smi_clean', 'atom_types', 'max_penalty', 'orig_embedding', 'clean_embedding', 'atomtype_to_embedding', 'atomtype_to_clean_selfies_embedding'])
    
    unique_atomtype_set = set(chain.from_iterable(merged_dikt[key]['atom_types'] for key in merged_dikt if merged_dikt[key].get('atom_types') is not None))
    atomtypes_to_elems_dict = create_elementsubsets(unique_atomtype_set)

    # get colors for atomtypes by element and element groups
    colordict = colorstoatomtypesbyelement(atomtypes_to_elems_dict)

    # plot embeddings
    min_dist = 0.1
    n_neighbors = 15
    alpha = 0.8
    penalty_threshold = 300
    save_path_prefix = f"./27Sept_delaney_bbbp_clearance_lipo_{valid_keys_count}_{model}_{traintype}_thresh{penalty_threshold}/"
    create_plotsperelem(merged_dikt, colordict, penalty_threshold, min_dist, n_neighbors, alpha, save_path_prefix)