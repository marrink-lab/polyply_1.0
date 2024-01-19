import networkx as nx
from polyply.src.big_smile_parsing import (res_pattern_to_meta_mol,
                                           force_field_from_fragments)
from polyply.src.map_to_molecule import MapToMolecule

def compatible(left, right):
    """
    Check bonding descriptor compatibility according
    to the BigSmiles syntax convetions.

    Parameters
    ----------
    left: str
    right: str

    Returns
    -------
    bool
    """
    if left == right:
        return True
    if left[0] == "<" and right[0] == ">":
        if left[1:] == right[1:]:
            return True
    if left[0] == ">" and right[0] == "<":
        if left[1:] == right[1:]:
            return True
    return False

def generate_edge(source, target, bond_type="bonding"):
    """
    Given a source and a target graph, which have bonding
    descriptors stored as node attributes, find a pair of
    matching descriptors and return the respective nodes.
    The function also returns the bonding descriptors. If
    no bonding descriptor is found an instance of LookupError
    is raised.

    Parameters
    ----------
    source: :class:`nx.Graph`
    target: :class:`nx.Graph`
    bond_type: `abc.hashable`
        under which attribute are the bonding descriptors
        stored.

    Returns
    -------
    ((abc.hashable, abc.hashable), (str, str))
        the nodes as well as bonding descriptors

    Raises
    ------
    LookupError
        if no match is found
    """
    source_nodes = nx.get_node_attributes(source, bond_type)
    target_nodes = nx.get_node_attributes(target, bond_type)
    for source_node in source_nodes:
        for target_node in target_nodes:
            bond_source = source_nodes[source_node]
            bond_target = target_nodes[target_node]
            if compatible(bond_source, bond_target):
                return ((source_node, target_node), (bond_source, bond_target))
    raise LookupError

class DefBigSmileParser:
    """
    Parse an a string instance of a defined BigSmile,
    which describes a polymer molecule.
    """

    def __init__(self):
        self.force_field = None
        self.meta_molecule = None
        self.molecule = None

    def edges_from_bonding_descrpt(self):
        """
        Make edges according to the bonding descriptors stored
        in the node attributes of meta_molecule residue graph.
        If a bonding descriptor is consumed it is set to None,
        however, the meta_molecule edge gets an attribute with the
        bonding descriptors that formed the edge.
        """
        for prev_node, node in nx.dfs_edges(self.meta_molecule):
            edge, bonding = generate_edge(self.meta_molecule.nodes[prev_node]['graph'],
                                          self.meta_molecule.nodes[node]['graph'])
            self.meta_molecule.nodes[prev_node]['graph'][edge[0]]['bonding'] = None
            self.meta_molecule.nodes[prev_node]['graph'][edge[1]]['bonding'] = None
            self.meta_molecule.molecule.add_edge(edge, bonding=bonding)

    def parse(self, big_smile_str):
        res_pattern, residues = big_smile_str.split('.')
        self.meta_molecule = res_pattern_to_meta_mol(res_pattern)
        self.force_field = force_field_from_fragments(residues)
        MapToMolecule(self.force_field).run_molecule(self.meta_molecule)
        self.edges_from_bonding_descrpt()
        return self.meta_molecule
