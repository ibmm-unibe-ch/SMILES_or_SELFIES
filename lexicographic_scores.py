"""Lexicographic distances
SMILES or SELFIES, 2023
"""
import re
from collections import defaultdict
from typing import Dict, List

import nltk
from rouge_score import rouge_scorer

from constants import PARSING_REGEX


def compute_damerau_levenshtein(sequence1: List[str], sequence2: List[str]) -> float:
    """Compute the Damerau Levenshtein distance between sequence1 and sequence2
    https://github.com/jamesturk/jellyfish/blob/d15fee2de05694fe65d1cbb78519f01955ec3154/jellyfish/_jellyfish.py#L133

    Args:
        sequence1 (List[str]): starting sequence
        sequence2 (List[str]): ending sequence

    Returns:
        float: Damerau Levenshtein distance
    """
    da = defaultdict(lambda: 0)
    maxdist = len(sequence1) + len(sequence2) + 2
    d = {}
    d[(0, 0)] = maxdist
    for i in range(len(sequence1) + 1):
        d[(i + 1, 0)] = maxdist
        d[(i + 1, 1)] = i
    for j in range(len(sequence2) + 1):
        d[(0, j + 1)] = maxdist
        d[(1, j + 1)] = j

    for i in range(1, len(sequence1) + 1):
        db = 0
        for j in range(1, len(sequence2) + 1):
            k = da[sequence2[j - 1]]
            l = db
            if sequence1[i - 1] == sequence2[j - 1]:
                cost = 0
                db = j
            else:
                cost = 1
            d[(i + 1, j + 1)] = min(
                d[(i, j)] + cost,
                d[(i + 1, j)] + 1,
                d[(i, j + 1)] + 1,
                d[(k, l)] + (i - k - 1) + 1 + (j - l - 1),
            )
        da[sequence1[i - 1]] = i
    return d[(len(sequence1) + 1, len(sequence2) + 1)]


def compute_levenshtein(sequence1: list[str], sequence2: list[str]) -> float:
    """Compute the Levenshtein distance between sequence1 and sequence2
    https://en.wikipedia.org/wiki/Levenshtein_distance

    Args:
        sequence1 (List[str]): starting sequence
        sequence2 (List[str]): ending sequence

    Returns:
        float: Levenshtein distance
    """
    v0 = list(range(len(sequence2) + 1))
    for i, _ in enumerate(sequence1):
        v1 = [i + 1]
        for j, _ in enumerate(sequence2):
            del_cost = v0[j + 1] + 1
            insert_cost = v0[j] + 1
            if sequence1[i] == sequence2[j]:
                sub_cost = v0[j]
            else:
                sub_cost = v0[j] + 1
            v1 = v1 + [min(del_cost, insert_cost, sub_cost)]
        v0 = v1
    return v0[-1]


def match_score(alpha: str, beta: str) -> int:
    """Trying to match two strings for Needleman Wunsch

    Args:
        alpha (str): first token
        beta (str): second token

    Returns:
        int: 1==match, else -1
    """
    if alpha == beta:
        return 1
    if alpha == "-" or beta == "-":
        return -1
    return -1


def compute_needleman_wunsch(sequence1: List[str], sequence2: List[str]) -> float:
    """Compute the Needleman Wunsch score between sequence1 and sequence2
    https://wilkelab.org/classes/SDS348/2019_spring/labs/lab13-solution.html

    Args:
        sequence1 (List[str]): starting sequence
        sequence2 (List[str]): ending sequence

    Returns:
        float: Needleman Wunsch score
    """
    n = len(sequence1)
    m = len(sequence2)
    if min(m, n) <= 0:
        return -(max(m, n))
    GAP_PENALTY = -1
    score = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        score[i][0] = GAP_PENALTY * i
    for j in range(n + 1):
        score[0][j] = GAP_PENALTY * j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score by checking the top, left, and diagonal cells
            match = score[i - 1][j - 1] + match_score(
                sequence1[j - 1], sequence2[i - 1]
            )
            delete = score[i - 1][j] + -1
            insert = score[i][j - 1] + -1
            # Record the maximum score from the three possible scores calculated above
            score[i][j] = max(match, delete, insert)

    return score[-1][-1]


def compute_rouge(input_str: List[str], output_str: List[str]) -> float:
    """Compute the ROUGE score between input_str and output_str

    Args:
        input_str (List[str]): starting sequence
        output_str (List[str]): ending sequence

    Returns:
        float: ROUGE score
    """
    rouge_metrics = ["rouge1", "rouge2", "rouge3", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_metrics, use_stemmer=False)
    scores = scorer.score(" ".join(input_str), " ".join(output_str))
    output_dict = {}
    for rouge_metric in rouge_metrics:
        output_dict[rouge_metric] = scores[rouge_metric][1]
    return output_dict


def compute_bleu(input_str: List[str], output_str: List[str]) -> float:
    """Compute the BLEU score between input_str and output_str

    Args:
        input_str (List[str]): starting sequence
        output_str (List[str]): ending sequence

    Returns:
        float: BLEU score
    """
    output_dict = {}
    output_dict["BLEU"] = nltk.translate.bleu_score.sentence_bleu(
        [input_str], output_str
    )
    output_dict["BLEU1"] = nltk.translate.bleu_score.sentence_bleu(
        [input_str], output_str, (1, 0, 0, 0)
    )
    output_dict["BLEU2"] = nltk.translate.bleu_score.sentence_bleu(
        [input_str], output_str, (0, 1, 0, 0)
    )
    output_dict["BLEU3"] = nltk.translate.bleu_score.sentence_bleu(
        [input_str], output_str, (0, 0, 1, 0)
    )
    output_dict["BLEU4"] = nltk.translate.bleu_score.sentence_bleu(
        [input_str], output_str, (0, 0, 0, 1)
    )
    return output_dict


def compute_distances(input_str: str, output_str: str) -> Dict[str, float]:
    """Compute all distances from input_str to output_str and return them in a dict

    Args:
        input_str (str): starting string
        output_str (str): output string

    Returns:
        Dict[str, float]: dict of lexicographic distances
    """
    input_tokens = [
        parsed_token.strip()
        for parsed_token in re.split(PARSING_REGEX, input_str)
        if parsed_token
    ]
    output_tokens = [
        parsed_token.strip()
        for parsed_token in re.split(PARSING_REGEX, output_str)
        if parsed_token
    ]
    output_dict = {}
    max_length = max(len(input_tokens), len(output_tokens))
    length_diff = abs(len(input_tokens) - len(output_tokens))
    output_dict["input_set"] = set(input_tokens)
    output_dict["output_set"] = set(output_tokens)
    output_dict["max_len"] = max_length
    output_dict["len_diff"] = length_diff
    output_dict["nw"] = compute_needleman_wunsch(input_tokens, output_tokens)
    output_dict["nw_norm"] = output_dict["nw"] / max_length
    output_dict["lev"] = compute_levenshtein(input_tokens, output_tokens)
    output_dict["lev_norm"] = output_dict["lev"] / max_length
    output_dict["dl"] = compute_damerau_levenshtein(input_tokens, output_tokens)
    output_dict["dl_norm"] = output_dict["dl"] / max_length
    output_dict = (
        output_dict
        | compute_bleu(input_tokens, output_tokens)
        | compute_rouge(input_tokens, output_tokens)
    )
    return output_dict
