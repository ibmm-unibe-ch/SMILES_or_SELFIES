import pandas as pd
from pathlib import Path
from ast import literal_eval
from typing import Dict, List, Any
from utils import unpickle


def get_elem(string: str) -> str:
    """
    Extract the first uppercase alphanumeric character from a token.

    Args:
        string (str): Input token string.

    Returns:
        str: The first alphanumeric character in uppercase.
    """
    return ''.join(filter(str.isalnum, string.upper()))[0]


def _load_embeddings(path: Path, prefix: str, index: str) -> pd.DataFrame:
    """
    Load embeddings from a CSV file and set the specified index.

    Args:
        path (Path): Path to the dataset directory.
        prefix (str): Filename suffix (without "embeds_").
        index (str): Column name to use as DataFrame index.

    Returns:
        pd.DataFrame: DataFrame with embeddings indexed by SMILES or SELFIES.
    """
    return pd.read_csv(path / f"embeds_{prefix}.csv").set_index(index)


def _literal_eval_row(df: pd.DataFrame, key: str, column: str = "embedding") -> List[Any]:
    """
    Retrieve and parse an embedding row from a DataFrame.

    Args:
        df (pd.DataFrame): Embedding DataFrame.
        key (str): Row key (SMILES or SELFIES string).
        column (str, optional): Column name storing embeddings. Defaults to "embedding".

    Returns:
        List[Any]: Parsed embedding as a nested list/array structure.
    """
    return literal_eval(df.loc[key][column])


def _expand_embedding(embeddings: List[Any], prefix: str, pos: int) -> Dict[str, float]:
    """
    Expand a single token embedding into a flattened dictionary.

    Args:
        embeddings (List[Any]): Embedding list from CSV.
        prefix (str): Prefix for output keys (e.g., "SMILES_BART_emb").
        pos (int): Position index for the embedding.

    Returns:
        Dict[str, float]: Flattened embedding with indexed keys.
    """
    return {f"{prefix}_{i}": val for i, val in enumerate(embeddings[pos][0])}


def construct_GAFF_dataset(path: Path, kekulised: bool = False) -> None:
    """
    Construct the GAFF dataset by combining SMILES and optionally SELFIES embeddings.

    Args:
        path (Path): Path to the dataset directory.
        kekulised (bool, optional): If True, exclude SELFIES embeddings. Defaults to False.

    Outputs:
        Saves "combined_pretraining_annotations.csv" in the given path.
    """
    # Load SMILES embeddings
    smiles_bart = _load_embeddings(path, "smiles_pretrained_BART_26_8", "SMILES")
    smiles_roberta = _load_embeddings(path, "smiles_pretrained_roberta_26_8", "SMILES")
    smiles_untrained = _load_embeddings(path, "smiles_pretrained_untrained_27_8", "SMILES")

    mapping = unpickle(path / "smilestoatomtypestoselfies_dikt_22_8.pkl")

    # Optionally load SELFIES embeddings
    selfies_bart = selfies_roberta = selfies_untrained = None
    if not kekulised:
        selfies_bart = _load_embeddings(path, "selfies_pretrained_BART_26_8", "SELFIES")
        selfies_roberta = _load_embeddings(path, "selfies_pretrained_roberta_26_8", "SELFIES")
        selfies_untrained = _load_embeddings(path, "selfies_pretrained_untrained_27_8", "SELFIES")

    rows = []
    for smiles, smiles_info in mapping.items():
        # Retrieve SMILES embeddings
        smiles_bart_emb = _literal_eval_row(smiles_bart, smiles)
        smiles_roberta_emb = _literal_eval_row(smiles_roberta, smiles)
        smiles_untrained_emb = _literal_eval_row(smiles_untrained, smiles)

        # Retrieve SELFIES embeddings if needed
        if not kekulised:
            selfies = smiles_info["selfies"]
            selfies_bart_emb = _literal_eval_row(selfies_bart, selfies)
            selfies_roberta_emb = _literal_eval_row(selfies_roberta, selfies)
            selfies_untrained_emb = _literal_eval_row(selfies_untrained, selfies)
            inverted_atom_dict = {val: key[0] for key, val in smiles_info["selfies_map"].items()}

        for pos_it, smiles_pos in enumerate(smiles_info['posToKeep']):
            curr_row: Dict[str, Any] = {
                "SMILES": smiles,
                "SMILES_pos": smiles_pos,
                "smiles_token": smiles_info["smi_clean"][pos_it],
                "label": smiles_info["atom_types"][pos_it],
            }

            # Add SMILES embeddings
            curr_row |= _expand_embedding(smiles_bart_emb, "SMILES_BART_emb", smiles_pos)
            curr_row |= _expand_embedding(smiles_roberta_emb, "SMILES_roberta_emb", smiles_pos)
            curr_row |= _expand_embedding(smiles_untrained_emb, "SMILES_untrained_emb", smiles_pos)

            # Add SELFIES embeddings if not kekulised
            if not kekulised:
                selfies_pos = inverted_atom_dict[smiles_pos]
                curr_row |= _expand_embedding(selfies_bart_emb, "SELFIES_BART_emb", selfies_pos)
                curr_row |= _expand_embedding(selfies_roberta_emb, "SELFIES_roberta_emb", selfies_pos)
                curr_row |= _expand_embedding(selfies_untrained_emb, "SELFIES_untrained_emb", selfies_pos)

            rows.append(curr_row)

    df = pd.DataFrame(rows)
    df["Element"] = df["smiles_token"].str.lstrip("[").str.upper().str.slice(0, 1)
    df.to_csv(path / "combined_pretraining_annotations.csv", index=False)


def construct_ETH_dataset(path: Path) -> None:
    """
    Construct the ETH dataset by aligning SMILES and SELFIES embeddings
    and filtering inconsistent molecules.

    Args:
        path (Path): Path to the dataset directory.

    Outputs:
        Saves "merged_eth_dataset.csv" in the given path.
    """
    # Load embeddings
    smiles_bart = _load_embeddings(path, "smiles_pretrained_BART_26_8", "SMILES")
    smiles_roberta = _load_embeddings(path, "smiles_pretrained_roberta_26_8", "SMILES")
    selfies_bart = _load_embeddings(path, "selfies_pretrained_BART_26_8", "SELFIES")
    selfies_roberta = _load_embeddings(path, "selfies_pretrained_roberta_26_8", "SELFIES")

    eth_mapping = pd.read_csv(path / "ETH_extended.csv")

    rows = []
    removed_mols = kept_mols = 0

    grouped = eth_mapping[[
        "SMILES", "selfies", "selfies_toks", "selfies_map",
        "tokenized_SMILES", "cleaned_tokenized_SMILES_pos"
    ]].groupby("SMILES").agg("first")

    for smiles, row in grouped.iterrows():
        fail = False
        inverted_selfies= {
            val: key[0] for key, val in literal_eval(row["selfies_map"]).items()
        }

        tokenized_smiles = literal_eval(row["tokenized_SMILES"])
        tokenized_selfies = literal_eval(row["selfies_toks"])

        smiles_bart_emb = _literal_eval_row(smiles_bart, smiles)
        smiles_roberta_emb = _literal_eval_row(smiles_roberta, smiles)

        selfies = row["selfies"]
        selfies_bart_emb = _literal_eval_row(selfies_bart, selfies)
        selfies_roberta_emb = _literal_eval_row(selfies_roberta, selfies)

        curr_rows = []
        for atom_idx, smiles_pos in enumerate(literal_eval(row["cleaned_tokenized_SMILES_pos"])):
            if smiles_pos not in inverted_selfies:
                removed_mols += 1
                fail = True
                break

            selfies_pos = inverted_selfies[smiles_pos]
            smiles_tok = tokenized_smiles[smiles_pos]
            selfies_tok = tokenized_selfies[selfies_pos]
            element = get_elem(smiles_tok)

            if element != get_elem(selfies_tok):
                removed_mols += 1
                fail = True
                break

            curr = {
                "SMILES": smiles,
                "atom_idx": atom_idx,
                "element": element,
                "smiles_pos": smiles_pos,
                "selfies_pos": selfies_pos,
                "SMILES_tok": smiles_tok,
                "SELFIES_tok": selfies_tok,
            }

            curr |= _expand_embedding(smiles_bart_emb, "SMILES_BART_emb", smiles_pos)
            curr |= _expand_embedding(smiles_roberta_emb, "SMILES_roberta_emb", smiles_pos)
            curr |= _expand_embedding(selfies_bart_emb, "SELFIES_BART_emb", selfies_pos)
            curr |= _expand_embedding(selfies_roberta_emb, "SELFIES_roberta_emb", selfies_pos)

            curr_rows.append(curr)

        kept_mols += 1
        if not fail:
            rows.extend(curr_rows)

    print(f"kept mols: {kept_mols}, removed mols: {removed_mols}")

    merged = pd.merge(eth_mapping, pd.DataFrame(rows), on=["SMILES", "atom_idx"])
    emb_cols = [col for col in merged.columns if "emb" in col]

    result = merged[
        ~(merged.DASH_IDX.isin(merged[merged.element_x != merged.element_y].DASH_IDX.unique()))
    ][["element_x", "mulliken", "resp1", "resp2", "dual",
       "mbis_dipole_strength", "SMILES"] + emb_cols]

    result.to_csv(path / "merged_eth_dataset.csv", index=False)
