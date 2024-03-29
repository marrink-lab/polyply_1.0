import re
import networkx as nx
import pysmiles
import vermouth
from polyply.src.big_smile_parsing import (res_pattern_to_meta_mol,
                                           force_field_from_fragments)
from polyply.src.map_to_molecule import MapToMolecule

VALENCES = pysmiles.smiles_helper.VALENCES
VALENCES.update({"H":(1,)})

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
    if left == right and left not in '> <':
        return True
    l, r = left[0], right[0]
    if (l, r) == ('<', '>') or (l, r) == ('>', '<'):
        return left[1:] == right[1:]
    return False

def generate_edge(source, target, bond_attribute="bonding"):
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
    bond_attribute: `abc.hashable`
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
    source_nodes = nx.get_node_attributes(source, bond_attribute)
    target_nodes = nx.get_node_attributes(target, bond_attribute)
    for source_node in source_nodes:
        for target_node in target_nodes:
            #print(source_node, target_node)
            bond_sources = source_nodes[source_node]
            bond_targets = target_nodes[target_node]
            for bond_source in bond_sources:
                for bond_target in bond_targets:
                    #print(bond_source, bond_target)
                    if compatible(bond_source, bond_target):
                        return ((source_node, target_node), (bond_source, bond_target))
    raise LookupError

class DefBigSmileParser:
    """
    Parse an a string instance of a defined BigSmile,
    which describes a polymer molecule.
    """

    def __init__(self, force_field):
        self.force_field = force_field
        self.meta_molecule = None
        self.molecule = None

    def edges_from_bonding_descrpt(self):
        """
        Make edges according to the bonding descriptors stored
        in the node attributes of meta_molecule residue graph.
        If a bonding descriptor is consumed it is removed from the list,
        however, the meta_molecule edge gets an attribute with the
        bonding descriptors that formed the edge. Later uncomsumed
        bonding descriptors are replaced by hydrogen atoms.
        """
        for prev_node, node in nx.dfs_edges(self.meta_molecule):
            prev_graph = self.meta_molecule.nodes[prev_node]['graph']
            node_graph = self.meta_molecule.nodes[node]['graph']
            edge, bonding = generate_edge(prev_graph,
                                          node_graph)
            # this is a bit of a workaround because at this stage the
            # bonding list is actually shared between all residues of
            # of the same type; so we first make a copy then we replace
            # the list sans used bonding descriptor
            prev_bond_list = prev_graph.nodes[edge[0]]['bonding'].copy()
            prev_bond_list.remove(bonding[0])
            prev_graph.nodes[edge[0]]['bonding'] = prev_bond_list
            node_bond_list = node_graph.nodes[edge[1]]['bonding'].copy()
            node_bond_list.remove(bonding[1])
            node_graph.nodes[edge[1]]['bonding'] = node_bond_list
            order = re.findall("\d+\.\d+", bonding[0])
            # bonding descriptors are assumed to have bonding order 1
            # unless they are specifically annotated
            if not order:
                order = 1
            self.meta_molecule.molecule.add_edge(edge[0], edge[1], bonding=bonding, order=order)

    def replace_unconsumed_bonding_descrpt(self):
        """
        We allow multiple bonding descriptors per atom, which
        however, are not always consumed. In this case the left
        over bonding descriptors are replaced by hydrogen atoms.
        """
        for meta_node in self.meta_molecule.nodes:
            graph = self.meta_molecule.nodes[meta_node]['graph']
            bonding = nx.get_node_attributes(graph, "bonding")
            for node, bondings in bonding.items():
                element = graph.nodes[node]['element']
                bonds = round(sum([self.meta_molecule.molecule.edges[(node, neigh)]['order'] for neigh in\
                                   self.meta_molecule.molecule.neighbors(node)]))
                hcount = VALENCES[element][0] - bonds + 1
                attrs = {attr: graph.nodes[node][attr] for attr in ['resname', 'resid', 'charge_group']}
                attrs['element'] = 'H'
                for new_id in range(1, hcount):
                    new_node = len(self.meta_molecule.molecule.nodes) + 1
                    graph.add_edge(node, new_node)
                    attrs['atomname'] = "H" + str(len(graph.nodes)-1)
                    graph.nodes[new_node].update(attrs)
                    self.meta_molecule.molecule.add_edge(node, new_node, order=1)
                    self.meta_molecule.molecule.nodes[new_node].update(attrs)
        # now we want to sort the atoms
        vermouth.SortMoleculeAtoms().run_molecule(self.meta_molecule.molecule)
        # and redo the meta molecule
        self.meta_molecule.relabel_and_redo_res_graph(mapping={})

    def parse(self, big_smile_str):
        res_pattern, residues = re.findall(r"\{[^\}]+\}", big_smile_str)
        self.meta_molecule = res_pattern_to_meta_mol(res_pattern)
        self.force_field = force_field_from_fragments(residues)
        MapToMolecule(self.force_field).run_molecule(self.meta_molecule)
        self.edges_from_bonding_descrpt()
        self.replace_unconsumed_bonding_descrpt()
        return self.meta_molecule

# ToDo
# - clean copying of bond-list attributes L100
