import re
from collections import defaultdict

from constants import PARSING_REGEX


def compute_damerau_levenshtein(sequence1, sequence2):
    # https://github.com/jamesturk/jellyfish/blob/d15fee2de05694fe65d1cbb78519f01955ec3154/jellyfish/_jellyfish.py#L133
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


def compute_levenshtein(sequence1: list[str], sequence2: list[str]):
    # https://en.wikipedia.org/wiki/Levenshtein_distance
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


def match_score(alpha, beta):
    if alpha == beta:
        return 1
    if alpha == "-" or beta == "-":
        return -1
    return -1


def compute_needleman_wunsch(sequence1, sequence2):
    # https://wilkelab.org/classes/SDS348/2019_spring/labs/lab13-solution.html
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


def compute_distances(input_str, output_str):
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
    output_dict["max_len"] = max_length
    output_dict["len_diff"] = length_diff
    output_dict["nw"] = compute_needleman_wunsch(input_tokens, output_tokens)
    output_dict["nw_norm"] = output_dict["nw"] / max_length
    output_dict["lev"] = compute_levenshtein(input_tokens, output_tokens)
    output_dict["lev_norm"] = output_dict["lev"] / max_length
    output_dict["dl"] = compute_damerau_levenshtein(input_tokens, output_tokens)
    output_dict["dl_norm"] = output_dict["dl"] / max_length
    return output_dict
