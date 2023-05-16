def get_atom(smiles: str, print_messages: bool = True) -> tuple(str, str):
    if len(smiles) == 0:
        return None, ""
    if print_messages:
        print(f"get_atom: {smiles}")
    if smiles[0].isalpha():
        return smiles[0], smiles[1:]
    if smiles[0] in [
        ".",
        "-",
        "=",
        "#",
        "$",
        ":",
        "/",
        "\\",
    ]:
        # from https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
        bond = smiles[0]
        atom, smiles = get_atom(smiles[1:], print_messages)
        return bond + atom, smiles
    if smiles[0] == "[":
        end = smiles.index("]")
        return smiles[0 : end + 1], smiles[end + 1 :]
    if smiles[0].isnumeric():
        return get_atom(smiles[1:], print_messages)
    # TODO throw exception?
    return None, smiles


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


def find_branch_end(smiles: str) -> tuple(str, str):
    if len(smiles) == 0 or smiles[0] != "(":
        return "", smiles
    branch = ""
    open_bracks = 0
    for index, char in enumerate(smiles):
        starting_index = index
        if char == ")":
            open_bracks -= 1
        elif char == "(":
            open_bracks += 1
        branch += char
        if open_bracks == 0:
            break
    return branch, smiles[starting_index + 1 :]


def deal_inner_ring(ring_smile, curr_id, last_ring_char, print_messages: bool = True):
    overlap = ""
    if ring_smile[0] in ring_smile[1:]:  # nested
        inner_ring_end = ring_smile[1:].index(ring_smile[0]) + 1
        post_append = (
            (curr_id, curr_id + 1),
            translate_ring(
                ring_smile[1:inner_ring_end],
                last_ring_char,
                ring_smile[0],
                print_messages,
            )[0],
        )
        ring_smile = (
            get_atom_end(post_append[1], print_messages)
            + ring_smile[inner_ring_end + 1 :]
        )
        if print_messages:
            print(f"inner ring post append {post_append}")
            print(f"updated ring SMILES: {ring_smile}")
    else:
        if print_messages:
            print(f"overlapstart: {ring_smile}")
        overlap = last_ring_char + ring_smile[0]
        ring_smile = ring_smile[1:]
        post_append = None
    return ring_smile, overlap, post_append


def parse_post(post: list, print_messages: bool = True) -> str:
    if print_messages:
        print(f"post:{post}")
    output = ""
    for post_item in post:
        printable = post_item[0]
        if print_messages:
            print(post_item[0])
        if isinstance(post_item[0], (tuple, list)):
            printable = ",".join([str(numb) for numb in post_item[0]])
        output += "?{" + str(printable) + "}" + post_item[1] + "!"
    return output


def parse_overlap(overlap_ids: list) -> str:
    if overlap_ids:
        ids = "{"
        for i in overlap_ids:
            if isinstance(i, (tuple, list)):
                i = ",".join(i)
            ids += str(i) + ","
        ids = ids[:-1] + "}"
        return ids
    return ""


def translate_ring(
    ring_smile: str, last_ring_char: str, ring_id: str, print_messages: bool = True
) -> tuple(str, str):
    overlap_ids = []
    curr_id = 0
    overlap = ""
    ring_string = last_ring_char
    post = []
    while len(ring_smile) > 0:
        if print_messages:
            print(f"ring_smile: {ring_smile}")
        if ring_smile[0].isnumeric() and ring_smile[0] != ring_id:  # overlap or nested
            ring_smile, overlap, post_append = deal_inner_ring(
                ring_smile, curr_id, last_ring_char, print_messages
            )
            if post_append is None:  # maybe is not None
                continue
            post.append(post_append)
        elif ring_smile[0] == "(":
            branch, ring_smile = find_branch_end(ring_smile)
            post.append((curr_id, translate_to_own(branch[1:-1], print_messages)))
            continue
        last_ring_char, ring_smile = get_atom(ring_smile, print_messages)
        ring_string += last_ring_char
        curr_id += 1
        if overlap:
            overlap += last_ring_char
            overlap_ids.append(curr_id)
    output_append = "<" + ring_string + ">"
    output_append += parse_post(post, print_messages)
    return output_append + parse_overlap(overlap_ids), overlap


def order_starts(curr_smile, print_messages: bool = True) -> tuple(str, int):
    ring_starts = []
    for curr_index, char in enumerate(curr_smile):
        if char.isnumeric():  # other ring starts
            start_index = (
                curr_smile[curr_index + 1 :].index(curr_smile[curr_index])
                + curr_index
                + 1
            )
            ring_starts.append((char, start_index))
        else:
            break
    starts = sorted(ring_starts, key=lambda tup: tup[1], reverse=True)
    curr_starts = "".join([start[0] for start in starts])
    curr_smile = curr_starts + curr_smile[len(starts) :]
    if print_messages:
        print("order_starts")
        print(curr_smile)
        print(ring_starts)
    return curr_smile, starts[0][1]


def order_ends(curr_smile: str, end_offset: int, print_messages: bool = True) -> str:
    pre = []
    post = []
    for char in curr_smile[end_offset:]:
        if char.isnumeric():
            if (
                char in curr_smile[1:end_offset]
            ):  # TODO does not support if Ring opened and closed
                pre.append(char)
            else:
                post.append(char)
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


def deal_ring(curr_smile, curr_atom, print_messages: bool = True) -> tuple(str, str):
    curr_smile, end_offset = order_starts(curr_smile, print_messages)
    curr_smile = order_ends(curr_smile, end_offset, print_messages)
    ring_id = curr_smile[0]
    if print_messages:
        print(f"ring {ring_id}, {curr_smile}")
        print(f"updated string: {curr_smile}")
    ring_end = curr_smile[1:].index(ring_id) + 1
    output_append, overlap = translate_ring(
        curr_smile[1:ring_end], curr_atom, ring_id, print_messages
    )
    if print_messages:
        print(f"overlap:{overlap}")
    return output_append, overlap + curr_smile[ring_end + 1 :]


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
        elif curr_smile[0].isnumeric():  # ring
            output_append, curr_smile = deal_ring(curr_smile, curr_atom, print_messages)
            output += output_append
        elif curr_smile[0] == "(":
            branch, curr_smile = find_branch_end(curr_smile)
            output += (
                curr_atom + "(" + translate_to_own(branch[1:-1], print_messages) + ")"
            )
        else:
            output += curr_atom
    return output
