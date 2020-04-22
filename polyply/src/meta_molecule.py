from collections import namedtuple
import json
import networkx as nx
from networkx.readwrite import json_graph
from vermouth.graph_utils import make_residue_graph
from polyply.src.parsers import read_polyply
from .polyply_parser import read_polyply

Monomer = namedtuple('Monomer', 'resname, n_blocks')


def _make_edges(force_field):
   for block in force_field.blocks.values():
       inter_types = list(block.interactions.keys())
       for inter_type in inter_types:
           block.make_edges_from_interaction_type(type_=inter_type)

   for link in force_field.links:
       inter_types = list(link.interactions.keys())
       for inter_type in inter_types:
           block.make_edges_from_interaction_type(type_=inter_type)

class MetaMolecule(nx.Graph):
    """
    Graph that describes molecules at the residue level.
    """

    def __init__(self, *args, **kwargs):
        self.force_field = kwargs.pop('force_field', None)
        super().__init__(*args, **kwargs)
        self.molecule = None

    def add_monomer(self, current, resname, connections):
        """
        This method adds a single node and an unlimeted number
        of edges to an instance of :class::`MetaMolecule`. Note
        that matches may only refer to already existing nodes.
        But connections can be an empty list.
        """
        self.add_node(current, resname=resname)
        for edge in connections:
            if self.has_node(edge[0]) and self.has_node(edge[1]):
                self.add_edge(edge[0], edge[1])
            else:
                msg = ("Edge {} referes to nodes that currently do"
                       "not exist. Cannot add edge to unkown nodes.")
                raise IOError(msg.format(edge))

    def get_edge_resname(self, edge):
        return [self.nodes[edge[0]]["resname"],  self.nodes[edge[1]]["resname"]]

    @staticmethod
    def _block_graph_to_res_graph(block):
        """
        generate a residue graph from the nodes of `block`.

        Parameters
        -----------
        block: `:class:vermouth.molecule.Block`

        Returns
        -------
        :class:`nx.Graph`
        """
        res_graph = make_residue_graph(block, attrs=('resid', 'resname'))
        return res_graph

    @classmethod
    def from_monomer_seq_linear(cls, force_field, monomers, mol_name):
        """
        Constructs a meta graph for a linear molecule
        which is the default assumption from
        """

        meta_mol_graph = cls(force_field=force_field, name=mol_name)
        res_count = 0

        for monomer in monomers:
            trans = 0
            while trans < monomer.n_blocks:

                if res_count != 0:
                    connect = [(res_count-1, res_count)]
                else:
                    connect = []
                trans += 1

                meta_mol_graph.add_monomer(res_count, monomer.resname, connect)
                res_count += 1
        return meta_mol_graph

    @classmethod
    def from_json(cls, force_field, json_file, mol_name):
        """
        Constructs a :class::`MetaMolecule` from a json file
        using the networkx json package.
        """
        with open(json_file) as file_:
            data = json.load(file_)

        graph = json_graph.node_link_graph(data)
        meta_mol = cls(graph, force_field=force_field, mol_name=mol_name)
        return meta_mol

    @classmethod
    def from_itp(cls, force_field, itp_file, mol_name):
        """
        Constructs a :class::`MetaMolecule` from an itp file.
        """
        with open(itp_file) as file_:
            lines = file_.readlines()
            read_polyply(lines, force_field)

        graph = MetaMolecule._block_graph_to_res_graph(force_field.blocks[mol_name])
        meta_mol = cls(graph, force_field=force_field, mol_name=mol_name)
        meta_mol.molecule = force_field.blocks[mol_name].to_molecule()
        return meta_mol

    @classmethod
    def from_block(cls, force_field, block, mol_name):
       """
       Constructs a :class::`MetaMolecule` from an vermoth.molecule.
       """
       _make_edges(force_field)
       graph = MetaMolecule._block_graph_to_res_graph(block)
       meta_mol = cls(graph, force_field=force_field, mol_name=mol_name)
       meta_mol.molecule = force_field.blocks[mol_name].to_molecule()
       return meta_mol
