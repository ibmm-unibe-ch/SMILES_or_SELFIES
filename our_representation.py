import re
from typing import List

from constants import OUR_REGEX

BONDS = [".", "-", "=", "#", "$", ":", "/", "\\"]


def get_atom(smiles: str, print_messages: bool = True) -> tuple[str, str]:
    if print_messages:
        print(f"get_atom: {smiles}")
    if len(smiles) == 0:
        return None, ""
    if smiles[0].isalpha():
        return smiles[0], smiles[1:]
    if smiles[0] in BONDS:
        # from https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
        bond = smiles[0]
        atom, smiles = get_atom(smiles[1:], print_messages)
        return bond + atom, smiles
    if smiles[0] == "[":
        end = smiles.index("]")
        return smiles[0 : end + 1], smiles[end + 1 :]
    if smiles[0].isnumeric():
        return get_atom(smiles[1:], print_messages)
    if smiles[0] == "%":
        return get_atom(smiles[3:], print_messages)
    raise Exception("Tried to get atom from empty/indifferent SMILES", smiles)


def get_atoms(smiles, print_messages=True):
    output = []
    next_atom = True
    while next_atom:
        next_atom, smiles = get_atom(smiles, print_messages)
        if next_atom:
            output.append(next_atom)
    return output


def shave_bond(smiles):
    if smiles and smiles[0] in BONDS:
        return shave_bond(smiles[1:])
    return smiles


def get_atom_end(smiles: str, print_messages: bool = True) -> str:
    counter = 1
    output = None
    corner_bracks = 0
    round_bracks = 0
    rings = 0
    while output is None:
        if smiles[-counter] == "]":
            corner_bracks += 1
        elif smiles[-counter] == "[":
            corner_bracks = max(0, corner_bracks - 1)
        elif smiles[-counter] == ")":
            round_bracks += 1
        elif smiles[-counter] == "(":
            round_bracks = max(0, round_bracks - 1)
        elif smiles[-counter] == "!":
            rings += 1
        elif smiles[-counter] == "?":
            rings = max(0, rings - 1)
        if rings == 0 and corner_bracks == 0 and round_bracks == 0:
            output, _ = get_atom(smiles[-counter:], print_messages)
        counter += 1
    return output


def find_branch_end(smiles: str) -> tuple[str, str]:
    if len(smiles) == 0:
        return "", smiles
    branch = "" if smiles[0] == "(" else smiles[0]
    open_bracks = 1
    for index, char in enumerate(smiles[1:]):
        starting_index = index
        if char == ")":
            open_bracks -= 1
        elif char == "(":
            open_bracks += 1
        branch += char
        if open_bracks == 0:
            break
    if open_bracks > 0:
        return "", smiles
    return branch, smiles[starting_index + 2 :]


def deal_inner_ring(ring_smile, curr_id, last_ring_char, print_messages: bool = True):
    overlap = ""
    ring_id = ring_smile[:3] if ring_smile[0] == "%" else ring_smile[0]
    len_ring_id = len(ring_id)
    if ring_id in ring_smile[len_ring_id:]:  # nested
        inner_ring_end = ring_smile[len_ring_id:].index(ring_id) + len_ring_id
        post_append_smiles, _, last_atom = translate_ring(
            ring_smile[len_ring_id:inner_ring_end],
            last_ring_char,
            ring_id,
            print_messages,
        )
        post_append = (
            (curr_id, curr_id + 1),
            post_append_smiles,
        )
        ring_smile = last_atom + ring_smile[inner_ring_end + len(ring_id) :]
        if print_messages:
            print(f"inner ring post append {post_append}")
            print(f"updated ring SMILES: {ring_smile}")
    else:
        if print_messages:
            print(f"overlapstart: {ring_smile}")
        overlap = last_ring_char + ring_id
        ring_smile = ring_smile[len_ring_id:]
        post_append = None
    return ring_smile, overlap, post_append


def sort_post_helper(ele, ring_start_index, ring_length):
    if isinstance(ele[0], (tuple, list)):
        return (ele[0][0] - ring_start_index) % ring_length
    return (ele[0] - ring_start_index) % ring_length


def parse_post(
    post: list, ring_start_index: int, ring_length: int, print_messages: bool = True
) -> str:
    if print_messages:
        print(f"post:{post}")
    post = sorted(
        post, key=lambda x: sort_post_helper(x, ring_start_index, ring_length)
    )
    output = ""
    for post_item in post:
        printable = post_item[0]
        if print_messages:
            print(post_item[0])
        if isinstance(post_item[0], (tuple, list)):
            numbers = [(numb - ring_start_index) % ring_length for numb in post_item[0]]
            printable = ",".join(
                ["%" + str(number) if number > 9 else str(number) for number in numbers]
            )
        elif isinstance(printable, int):
            number = (printable - ring_start_index) % ring_length
            printable = "%" + str(number) if number > 9 else str(number)
        output += "?{" + str(printable) + "}" + post_item[1] + "!"
    return output


def parse_overlap(overlap_ids: list) -> str:
    if overlap_ids:
        ids = "{"
        for i in overlap_ids:
            if isinstance(i, (tuple, list)):
                i = ",".join(["%" + str(elem) if elem > 9 else str(elem) for elem in i])
            ids += str("%" + str(i) if isinstance(i, int) and i > 9 else str(i)) + ","
        ids = ids[:-1] + "}"
        return ids
    return ""


def canonize_ring(ring: str, print_messages):
    ring = ring.strip()
    parsed_ring = get_atoms(ring, print_messages=print_messages)
    len_ring = len(parsed_ring)
    ring_ring = parsed_ring + parsed_ring
    possible_orders = [ring_ring[start : start + len_ring] for start in range(len_ring)]
    best_order = max(possible_orders)
    index_best = possible_orders.index(best_order)
    last_atom = parsed_ring[-1]
    return "".join(best_order), index_best, len_ring, last_atom


def translate_ring(
    ring_smile: str, last_ring_char: str, ring_id: str, print_messages: bool = True
) -> tuple[str, str]:
    overlap_ids = []
    curr_id = 0
    overlap = ""
    ring_string = last_ring_char
    post = []
    append_brack = ""
    while len(ring_smile) > 0:
        if print_messages:
            print(f"ring_smile: {ring_smile}")
        if (
            ring_smile[0].isnumeric() or ring_smile[0] == "%"
        ) and not ring_smile.startswith(
            ring_id
        ):  # overlap or nested
            ring_smile, overlap, post_append = deal_inner_ring(
                ring_smile, curr_id, last_ring_char, print_messages
            )
            if post_append is None:  # maybe is not None
                continue
            post.append(post_append)
        elif ring_smile[0] == "(":
            branch, ring_smile = find_branch_end(ring_smile)
            if branch:
                post.append(
                    (
                        curr_id,
                        translate_to_own(
                            shave_bond(last_ring_char) + branch[:-1], print_messages
                        ),
                    )
                )
            else:  # ring ends in branch
                ring_smile = ring_smile[1:]
                append_brack = "("
            continue
        last_ring_char, ring_smile = get_atom(ring_smile, print_messages)
        ring_string += last_ring_char
        curr_id += 1
        if overlap:
            overlap += last_ring_char
            overlap_ids.append(curr_id)
    canonized_ring_string, ring_start_index, len_ring, last_atom = canonize_ring(
        ring_string, print_messages
    )
    output_append = f"<{ring_start_index};{canonized_ring_string}>"
    output_append += parse_post(post, ring_start_index, len_ring, print_messages)
    return output_append + parse_overlap(overlap_ids) + append_brack, overlap, last_atom


def order_starts(curr_smile, print_messages: bool = True) -> tuple[str, int]:
    ring_starts = []
    drain = 0
    for curr_index, char in enumerate(curr_smile):
        if drain > 0:
            drain -= 1
        elif char.isnumeric() or char == "%":  # other ring starts
            curr_id = char
            if char == "%":
                drain = 2
                curr_id = curr_smile[curr_index : curr_index + 3]
            start_index = (
                curr_smile[curr_index + len(curr_id) :].index(curr_id) + curr_index + 1
            )
            ring_starts.append((curr_id, start_index))
        else:
            break
    starts = sorted(ring_starts, key=lambda tup: tup[1], reverse=True)
    curr_starts = "".join([start[0] for start in starts])
    curr_smile = curr_starts + curr_smile[len(curr_starts) :]
    if print_messages:
        print("order_starts")
        print(curr_smile)
        print(ring_starts)
    return curr_smile, starts[0][1] + len(starts[0][0]) - 1


def order_ends(curr_smile: str, end_offset: int, print_messages: bool = True) -> str:
    pre = []
    post = []
    drain = 0
    for curr_id, char in enumerate(curr_smile[end_offset:]):
        if drain > 0:
            drain -= 1
        elif char.isnumeric() or char == "%":
            curr_ring = char
            if char == "%":
                drain = 2
                curr_ring = curr_smile[end_offset + curr_id : end_offset + curr_id + 3]
            if curr_ring in curr_smile[1:end_offset]:
                pre.append(curr_ring)
            else:
                post.append(curr_ring)
        else:
            break
    curr_ends = "".join(pre) + "".join(post)
    if print_messages:
        print("end debug")
        print(curr_smile[:end_offset])
        print(curr_ends)
        print(curr_smile[end_offset + len(curr_ends) :])
    return (
        curr_smile[:end_offset] + curr_ends + curr_smile[end_offset + len(curr_ends) :]
    )


def deal_ring(curr_smile, curr_atom, print_messages: bool = True) -> tuple[str, str]:
    curr_smile, end_offset = order_starts(curr_smile, print_messages)
    curr_smile = order_ends(curr_smile, end_offset, print_messages)
    ring_id = curr_smile[:3] if curr_smile[0] == "%" else curr_smile[0]
    len_ring_id = len(ring_id)
    if print_messages:
        print(f"ring {ring_id}, {curr_smile}")
        print(f"updated string: {curr_smile}")
    offset = 1 if len_ring_id == 1 else 3
    ring_end = curr_smile[len_ring_id:].index(ring_id) + offset
    output_append, overlap, last_atom = translate_ring(
        curr_smile[len_ring_id:ring_end], curr_atom, ring_id, print_messages
    )
    if output_append.endswith("("):  # ring_end in branch
        last_open_brack = curr_smile[: ring_end + len(ring_id)][::-1].index("(")
        last_atom = get_atom_end(
            curr_smile[: ring_end + len(ring_id) - last_open_brack - 1], print_messages
        )
        ending_branch, return_smile = find_branch_end(
            curr_smile[ring_end + len(ring_id) :]
        )
        output_append = (
            "("
            + output_append[:-1]
            + translate_to_own(ending_branch[:-1], print_messages)
            + ")"
        )
    else:
        return_smile = curr_smile[ring_end + len(ring_id) :]
    if print_messages:
        print(f"overlap:{overlap}")
    return output_append, overlap + return_smile, last_atom


def check_if_unclosed_ring_in_branch(branch: str):
    drain = 0
    output = {}
    for branch_counter, char in enumerate(branch):
        if drain > 0:
            drain -= 1
        elif char.isnumeric() or char == "%":
            curr_id = (
                branch[branch_counter : branch_counter + 3] if char == "%" else char
            )
            if char == "%":
                drain = 2
            if output.pop(curr_id, True):
                output[curr_id] = False
    return False


def translate_to_own(curr_smile: str, print_messages: bool = True) -> str:
    output = ""
    while len(curr_smile) > 0:
        curr_atom, curr_smile = get_atom(curr_smile, print_messages)
        if print_messages:
            print(f"curr_smile:{curr_smile}")
            print(f"output: {output}")
        if curr_atom is None:
            curr_atom = ""
        if curr_smile == "":
            output += curr_atom
        elif curr_smile[0].isnumeric() or curr_smile[0] == "%":  # ring
            output_append, curr_smile, last_atom = deal_ring(
                curr_smile, curr_atom, print_messages
            )
            output += output_append
            if curr_smile:
                curr_smile = shave_bond(last_atom) + curr_smile
        elif curr_smile[0] == "(":
            branch, curr_smile = find_branch_end(curr_smile)
            if output[-1] == ")":
                starting_atom = ""
            else:
                starting_atom = curr_atom
            output += (
                starting_atom
                + "("
                + translate_to_own(shave_bond(curr_atom) + branch[:-1], print_messages)
                + ")"
            )
            if curr_smile:
                curr_smile = shave_bond(curr_atom) + curr_smile
        else:
            output += curr_atom

    return output


def tokenise_our_representation(our_representation: str) -> List[str]:
    return [token for token in re.split(OUR_REGEX, our_representation) if token]
