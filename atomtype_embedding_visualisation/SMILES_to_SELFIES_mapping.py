import selfies
from rdkit import Chem
import re

#from constants
PARSING_REGEX = r"(<unk>|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

#from preprocessing
def canonize_smiles(input_str: str, remove_identities: bool = True) -> str:
    """Canonize SMILES string

    Args:
        input_str (str): SMILES input string

    Returns:
        str: canonize SMILES string
    """
    mol = Chem.MolFromSmiles(input_str)
    if mol is None:
        return None
    # not sure remove_identities is neccessary for generate mapping, cannot see a difference
    if remove_identities:
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]

    return Chem.MolToSmiles(mol)

def generate_mapping(smiles, debug=False):
    # this mapping function is build on the basis that 
    # 1. the SELFIES string is generated from a canonized SMILES string, should work without canonization just the same
    # 2. the order of the atoms in both strings is the same
    # see: https://github.com/aspuru-guzik-group/selfies/blob/master/README.md 
    canon_smiles = canonize_smiles(smiles)
    assert smiles==canon_smiles, f"SMILES input was not canonized: {smiles} -> {canon_smiles}"
    
    selfies_str, attr = selfies.encoder(smiles, attribute=True)
    #print(attr)
    #attr is a list of AttributionMaps containing the output token, its index, and input tokens that led to it.
    # e.g AttributionMap(index=0, token='[O]', attribution=[Attribution(index=0, token='O')]),
    ind_list = [att.index for att in attr]
    index_dict = {elem: ind_list.count(elem) for elem in set(ind_list)}
    tokenised_smiles = [elem for elem in re.split(PARSING_REGEX,canonize_smiles(smiles)) if elem]
    tokenised_selfies = list(selfies.split_selfies(selfies_str))
    # Items map indices and tokens from the SELFIES representation to indices in the tokenized SMILES string. 
    # This mapping only includes tokens that do not represent rings or branches, occur exactly once (index_dict[att.index]==1), and match the corresponding token in the tokenized SELFIES string.
    # here att.index is the index of the token in the SELFIES string, att token is token in SELFIES string, and att.attribution[0].index is the index of the token in the SMILES string
    items = {(att.index, att.token):att.attribution[0].index for att in attr if (att.attribution and not (("Ring" in att.token) or ("Branch" in att.token)) and index_dict[att.index]==1 and att.token == tokenised_selfies[att.index])}
    if debug:
        print(smiles)
        print(selfies_str)
        print(attr)
        print(index_dict)
        print(items)
        for key, value in items.items():
            print(f"{list(selfies.split_selfies(selfies_str))[key[0]]} to {tokenised_smiles[value]}, {key} to {value}")
        print("missing SMILES")
        print([(val,tokenised_smiles[val]) for val in range(len(tokenised_smiles)) if not (val in list(items.values())) ] )
        print([(key,tokenised_selfies[key]) for key in range(len(tokenised_selfies)) if not (key in list(items.keys())) ] )
    else:
        # Checks that there are no duplicate indices in the SELFIES tokens part of the mapping. 
        # #This is done by comparing the length of the set of SELFIES indices (which removes duplicates) with the length of the list of these indices. If the lengths are equal, there are no duplicates.
        no_dupes_i = len(set([key[0] for key in items.keys()]))==len([key[0] for key in items.keys()])
        # Similar to no_dupes_i, but checks for no duplicate indices in the SMILES part of the mapping.
        no_dupes_ii = len(set([val for val in items.values()]))==len([val for val in items.values()])
        # Ensures there are no alphabetic SMILES tokens left unmapped. 
        # It checks if any alphabetic token in the tokenized SMILES string is not part of the mapping (items.values()), indicating a potentially incomplete mapping.
        no_leftover_smiles = not any([tokenised_smiles[val].isalpha() for val in range(len(tokenised_smiles)) if not (val in list(items.values()))])
        correct_letter = not any(["".join([v.upper() for v in tokenised_smiles[val] if v.isalpha() or v=="@"]) not in key[1] for (key, val) in list(items.items()) if any([v.isalpha() for v in tokenised_smiles[val]])])
        if no_dupes_i and no_dupes_ii and no_leftover_smiles and correct_letter:
            return selfies_str,tokenised_selfies,items
        else:
            return selfies_str,tokenised_selfies,None
        
def generate_mappings_for_task_SMILES_to_SELFIES(task_SMILES):
    mappings = {}
    for smiles in task_SMILES:
        selfies_str,tokenised_selfies,mapping = generate_mapping(smiles)
        mappings[smiles] = {}
        mappings[smiles]['selfiesstr_tok_map'] = (selfies_str,tokenised_selfies,mapping)
    return mappings