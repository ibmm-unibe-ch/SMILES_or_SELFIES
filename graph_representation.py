import re
from typing import List

import networkx as nx
from rdkit import Chem

OUR_REGEX = r"(<[0-9]+;|<unk>|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?\{|>>?|\*|\$|\%[0-9]{2}|[0-9]|\,|\!|\})"


def generate_edges_to_remove(node_ids):
    outputs = []
    for index, node_id in enumerate(node_ids[1:]):
        if node_id > node_ids[index]:
            start, end = node_ids[index], node_id
        else:
            start, end = node_id, node_ids[index]
        outputs.append((start, end))
    if len(node_ids) > 2:
        if node_ids[0] > node_ids[-1]:
            start, end = node_ids[-1], node_ids[0]
        else:
            start, end = node_ids[0], node_ids[-1]
        outputs.append((start, end))
    return set(outputs)


def generate_min_cycles(graph):
    output = dict()
    for cycle_nodes in nx.minimum_cycle_basis(graph):
        subgraph = graph.subgraph(cycle_nodes)
        len_cycle_nodes = len(cycle_nodes)
        for cycle in nx.simple_cycles(subgraph):
            if len_cycle_nodes == len(cycle):
                cycle_object = tuple(cycle)
                for node in cycle_object:
                    output[node] = output.get(node, {}) | {cycle_object: True}
            continue
    return output


BOND_DICT = {
    Chem.rdchem.BondType.SINGLE: "",
    Chem.rdchem.BondType.DOUBLE: "=",
    Chem.rdchem.BondType.TRIPLE: "#",
    Chem.rdchem.BondType.QUADRUPLE: "$",
    Chem.rdchem.BondType.AROMATIC: "",
}
CHIRAL_DICT = {
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: "@",
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: "@@",
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: "",
}


class Molecule_Graph:
    def __init__(self, input_smiles):
        self.done_cycles = {}
        self.done_nodes = {}
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            raise ValueError(f'"{input_smiles}" could not be parsed by RDKit.')
        self.canonised_smiles = Chem.MolToSmiles(mol)
        self.graph = self.create_graph(Chem.MolFromSmiles(self.canonised_smiles))
        self.nodes = set(tuple(self.graph.nodes))
        self.cycles = generate_min_cycles(self.graph)
        self.edges = set(tuple(self.graph.edges))

    def create_graph(self, graph):
        G = nx.Graph()
        for atom in graph.GetAtoms():
            G.add_node(
                atom.GetIdx(),
                hs=atom.GetNumExplicitHs(),
                chiral_tag=atom.GetChiralTag(),
                charge=atom.GetFormalCharge(),
                is_aromatic=atom.GetIsAromatic(),
                atom_symbol=atom.GetSymbol(),
            )
        for bond in graph.GetBonds():
            begin_index = bond.GetBeginAtomIdx()
            end_index = bond.GetEndAtomIdx()
            if begin_index > end_index:
                begin_index, end_index = end_index, begin_index
            G.add_edge(begin_index, end_index, bond_type=bond.GetBondType())
        return G

    def get_atom_symbol(self, node_id):
        node_info = self.graph.nodes[node_id]
        symbol = node_info["atom_symbol"]
        if node_info["is_aromatic"]:
            symbol = symbol.lower()
        try:
            symbol += CHIRAL_DICT[node_info["chiral_tag"]]
        except KeyError:
            raise KeyError(
                f'Chirality {node_info["chiral_tag"]} not found in chiral dict.'
            )
        hs = node_info["hs"]
        if hs:
            symbol = f"{symbol}H{hs}"
            if hs == 1:
                symbol = symbol[:-1]
        charge = node_info["charge"]
        if charge != 0:
            charge_char = "+" * charge if charge > 0 else "-" * charge
            symbol = symbol + charge_char
        return symbol if len(symbol) == 1 else f"[{symbol}]"

    def get_bond_symbol(self, start_index, end_index):
        if start_index < end_index:
            edge = self.graph.edges[start_index, end_index]
        else:
            edge = self.graph.edges[end_index, start_index]
        if edge:
            bond_type = edge["bond_type"]
            return BOND_DICT[bond_type]
        raise KeyError(f"Bond type {bond_type} not found in bond types.")

    def ring_to_string(self, node_ids):
        output = self.get_atom_symbol(node_ids[0])
        for index, curr_end_id in enumerate(node_ids[1:]):
            edge_symbol = self.get_bond_symbol(node_ids[index], curr_end_id)
            atom_symbol = self.get_atom_symbol(curr_end_id)
            output += edge_symbol + atom_symbol
        # last edge is not written, I think like in SMILES
        return output

    # check canonisation
    def canonize_ring(self, node_ids):
        ring_string = [self.get_atom_symbol(node_id) for node_id in node_ids]
        len_ring = len(node_ids)
        ring_ring = ring_string + ring_string
        possible_orders = [
            ring_ring[start : start + len_ring] for start in range(len_ring)
        ]
        best_order = min(possible_orders)
        index_best = possible_orders.index(best_order)
        node_ids_node_ids = node_ids + node_ids
        return node_ids_node_ids[index_best : index_best + len_ring]

    def process_ring(self, node_ids, start_id):
        self.edges = self.edges - generate_edges_to_remove(node_ids)
        canonised_ring_ids = self.canonize_ring(tuple(node_ids))
        ring_start = canonised_ring_ids.index(start_id)
        canonised_ring_string = self.ring_to_string(canonised_ring_ids)
        output = f"<{ring_start};{canonised_ring_string}>"
        todo_nodes = [node_id for node_id in node_ids if node_id in self.nodes]
        self.nodes = self.nodes - set(todo_nodes)
        additional_chains = []
        for node in todo_nodes:
            if node in self.cycles:
                cycles_node = list(self.cycles[node])
                for cycle in cycles_node:
                    if not (cycle in self.cycles[node]):
                        continue
                    for cycle_node in cycle:
                        del self.cycles[cycle_node][cycle]
                    overlap_ids = set(node_ids).intersection(cycle)
                    overlap_indices = [
                        str(canonised_ring_ids.index(overlap_id))
                        for overlap_id in overlap_ids
                    ]
                    sorted_overlap_indices = sorted(overlap_indices, key=int)
                    additional_chains.append((sorted_overlap_indices, self.process_ring(cycle,node)))
                    
            node_output = self.process_node(node)
            if node_output != self.get_atom_symbol(node):
                additional_chains.append(([str(canonised_ring_ids.index(node))],node_output))
        sorted_additional_chains = sorted(additional_chains, key=lambda x: int(x[0][0]))
        for sorted_additional_chain in sorted_additional_chains[:-1]:
            output += f"?{{{','.join(sorted_additional_chain[0])}}}{sorted_additional_chain[1]}!"
        if sorted_additional_chains:
            len_last_overlap = len(sorted_additional_chains[-1][0])
            if sorted_additional_chains[-1][0] == [str(elem) for elem in list(range(len(node_ids))[-len_last_overlap:])]: 
                output += f"{sorted_additional_chains[-1][1]}"
            else:
                output += f"?{{{','.join(sorted_additional_chains[-1][0])}}}{sorted_additional_chains[-1][1]}!"
        return output

    def process_node(self, node_id):
        if node_id in self.nodes:
            self.nodes.remove(node_id)
        curr_atom_symbol = self.get_atom_symbol(node_id)
        output = ""
        seen_cycle = False
        cycles = list(self.cycles.get(node_id, []))
        for cycle in cycles:
            if cycle in self.cycles[node_id]:
                seen_cycle = True
                for cycle_node in cycle:
                    del self.cycles[cycle_node][cycle]
                output += self.process_ring(cycle, (node_id))
        if seen_cycle:
            return output
        nexts = []
        for neighbour_id in list(self.graph.neighbors(node_id)):
            if neighbour_id < node_id:
                start_id, end_id = neighbour_id, node_id
            else:
                start_id, end_id = node_id, neighbour_id
            if (start_id, end_id) in self.edges:
                self.edges.remove((start_id, end_id))
                nexts.append(
                    self.get_bond_symbol(start_id, end_id)
                    + self.process_node(neighbour_id)
                )
        output = ""
        for nextt in nexts[:-1]:
            output += f"{curr_atom_symbol}({nextt})"
        if nexts:
            output += curr_atom_symbol + nexts[-1]
        else:
            output += curr_atom_symbol
        return output

    def process_graph(self):
        return self.process_node(0)


def translate_to_graph_representation(input_smiles):
    graph = Molecule_Graph(input_smiles)
    output = graph.process_graph()
    return output


def tokenise_our_representation(our_representation: str) -> List[str]:
    return [token for token in re.split(OUR_REGEX, our_representation) if token]
