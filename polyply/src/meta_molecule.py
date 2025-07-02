# Copyright 2020 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import (namedtuple, OrderedDict)
import networkx as nx
from pysmiles import PTE
from cgsmiles.resolve import MoleculeResolver
from cgsmiles.read_cgsmiles import read_cgsmiles
from vermouth.graph_utils import make_residue_graph
from vermouth.log_helpers import StyleAdapter, get_logger
from vermouth.gmx.itp_read import read_itp
from .graph_utils import find_nodes_with_attributes, find_one_ismags_match
from .simple_seq_parsers import parse_txt, parse_ig, parse_fasta, parse_json, Monomer

LOGGER = StyleAdapter(get_logger(__name__))

def _make_edges(force_field):
    for block in force_field.blocks.values():
        inter_types = list(block.interactions.keys())
        for inter_type in inter_types:
            if inter_type in ["bonds", "constraints"]:
                block.make_edges_from_interaction_type(type_=inter_type)

    for link in force_field.links:
        inter_types = list(link.interactions.keys())
        for inter_type in inter_types:
            if inter_type in ["bonds", "constraints", "angles"]:
                link.make_edges_from_interaction_type(type_=inter_type)

def _interpret_residue_mapping(graph, resname, new_residues):
    """
    Find all nodes corresponding to resname in graph
    and generate a corrspondance dict of these nodes
    to new resnames as defined by new_residues string
    which has the format <resname-atom1,atom2 ...>.

    Parameters:
    -----------
    graph: networkx.graph
    resname: str
    new_residues:  list[str]

    Returns:
    --------
    dict
        mapping of nodes to new residue name
    """
    atom_name_to_resname = {}
    had_atoms = []

    for new_res in new_residues:
        new_name, atoms = new_res.split("-")
        names = atoms.split(",")

        for name in names:
            if name in had_atoms:
                msg = ("You are trying to split residue {} into {} residues. "
                       "However, atom {} is mentioned more than once. This is not "
                       "allowed. ")
                raise IOError(msg.format(resname, len(new_residues), name))
            nodes = find_nodes_with_attributes(graph, resname=resname, atomname=name)
            had_atoms.append(name)

            for node in nodes:
                atom_name_to_resname[node] = new_name
    return atom_name_to_resname

def _find_starting_node(meta_molecule):
    """
    Find the first node that has coordinates if there is
    otherwise return first node in list of nodes.
    """
    for node in meta_molecule.nodes:
        if "build" not in meta_molecule.nodes[node]:
            return node
    return next(iter(meta_molecule.nodes()))

class MetaMolecule(nx.Graph):
    """
    Graph that describes molecules at the residue level.
    """

    node_dict_factory = OrderedDict
    parsers = { "txt": parse_txt,
               "fasta": parse_fasta,
               "ig": parse_ig,
               "json": parse_json,}

    def __init__(self, *args, **kwargs):
        self.force_field = kwargs.pop('force_field', None)
        self.mol_name = kwargs.pop('mol_name', None)
        super().__init__(*args, **kwargs)
        self.molecule = None
        nx.set_node_attributes(self, True, "build")
        nx.set_node_attributes(self, True, "backmap")
        self.__search_tree = None
        self.root = None
        self.dfs = False
        self.max_resid = 0
        self.__mass_to_element = None

        # add resids to polyply meta-molecule nodes if they are not
        # present. All algorithms rely on proper resids
        for node in self.nodes:
            if "resid" not in self.nodes[node]:
                LOGGER.warning("Node {} has no resid. Setting resid to {} + 1.", node, node)

                try:
                    self.nodes[node]["resid"] = node + 1
                except TypeError:
                    msg = "Couldn't add 1 to node. Either provide resids or use integers as node keys."
                    raise IOError(msg)

            if self.max_resid < self.nodes[node]["resid"]:
                self.max_resid = self.nodes[node]["resid"]

    def add_node(self, *args, **kwargs):
        self.max_resid += 1
        kwargs["resid"] = self.max_resid
        super().add_node(*args, **kwargs)

    def add_monomer(self, current, resname, connections):
        """
        This method adds a single node and an unlimeted number
        of edges to an instance of :class::`MetaMolecule`. Note
        that matches may only refer to already existing nodes.
        But connections can be an empty list.
        """
        self.add_node(current, resname=resname, build=True, backmap=True)
        for edge in connections:
            if self.has_node(edge[0]) and self.has_node(edge[1]):
                self.add_edge(edge[0], edge[1])
            else:
                msg = ("Edge {} referes to nodes that currently do"
                       "not exist. Cannot add edge to unkown nodes.")
                raise IOError(msg.format(edge))

    def get_edge_resname(self, edge):
        return [self.nodes[edge[0]]["resname"], self.nodes[edge[1]]["resname"]]

    def mass_to_element(self, mass):
        if self.__mass_to_element is None:
            self.__mass_to_element = {round(PTE[ele]['AtomicMass']): ele for ele in PTE if type(ele)==str}
        try:
            ele = self.__mass_to_element[round(mass)]
        except KeyError:
            raise IOError(f"Did not find element with mass {mass}.")
        return ele

    def set_element_from_mass(self, topology):
        """
        Set the element of an atom by matching its mass to the PTE.
        """
        for node in self.molecule.nodes:
            mass = self.molecule.nodes[node].get('mass',
                                                 topology.atom_types[self.molecule.nodes[node]['atype']])
            element = self.mass_to_element(mass)
            self.molecule.nodes[node]["element"] = element

    def relabel_and_redo_res_graph(self, mapping):
        """
        Relable the nodes of `self.molecule` using `mapping`
        and regenerate the meta_molecule (i.e. residue graph).

        Parameters:
        -----------
        mapping: dict
            mapping of node-key to new residue name
        """
        # find the maximum resiude id
        max_resid = self.max_resid
        # resname the residues and increase with pseudo-resid
        for node, resname in mapping.items():
            self.molecule.nodes[node]["resname"] = resname
            old_resid = self.molecule.nodes[node]["resid"]
            self.molecule.nodes[node]["resid"] = old_resid + max_resid
            self.molecule.nodes[node]["build"] = True
            self.molecule.nodes[node]["backmap"] = True

        # make a new residue graph and overwrite the old one
        new_meta_graph = make_residue_graph(self.molecule, attrs=('resid', 'resname'))

        # we need to do some bookkeeping for the resids
        for idx, node in enumerate(new_meta_graph.nodes):
            new_meta_graph.nodes[node]["resid"] = idx
            for atom in new_meta_graph.nodes[node]["graph"]:
                self.molecule.nodes[atom]["resid"] = idx

        self.clear()
        self.add_nodes_from(new_meta_graph.nodes(data=True))
        self.add_edges_from(new_meta_graph.edges)

    def split_residue(self, split_strings):
        """
        Split all residues defind by the in `split_strings`, which is a list
        of strings with format <resname>:<new_resname>-<atom1>,<atom2><etc>
        into new residues and also update the underlying molecule with these
        new residues.

        Parameters:
        -----------
        split_strings:  list[str]
             list of split strings

        Returns:
        dict
            mapping of the nodes to the new resnames
        """
        mapping = {}
        for split_string in split_strings:
            # split resname and new resiude definitions
            resname, *new_residues = split_string.split(":")
            # find out which atoms map to which new residues
            mapping.update(_interpret_residue_mapping(self.molecule, resname, new_residues))
        # relabel graph and redo residue graph
        self.relabel_and_redo_res_graph(mapping)
        return mapping

    def relabel_from_cgsmiles_str(self, cgsmiles_str, all_atom=False, topology=None):
        """
        Relabel the residue definition of a molecule from a cgsmiles string.

        As an example consider Martini POPC. Following the PEGylated lipid
        tutorial we want to relabel the lipid such that the head-group is
        one residue and the tails are one residue.

        >>> from polyply.src.meta_molecule import MetaMolecule
        >>> # as input we use a CGsmiles string of the lipid as whole
        >>> cgs = "{[#POPC]}.{#POPC=[#Q1][#Q5][#SN4a]([#C1][#C4h][#C1B][#C1])[#N4a][#C1][#C1][#C1][#C1]}"
        >>> mol = MetaMolecule.from_cgsmiles_str([], cgs, "test", seq_only=False)
        >>> # now we use a different cgsmiles string that has two coarse levels to
        >>> # relabel the molecule
        >>> new_cgs = "{[#HEAD][#TAIL]}.{#HEAD=[#Q1][#Q5][$],#TAIL=[$][#SN4a]([#C1][#C4h][#C1B][#C1])[#N4a][#C1][#C1][#C1][#C1]}"
        >>> mol.relabel_from_cgsmiles_str(new_cgs)

        Parameters
        ----------
        cgsmiles_str: str
            string in CGsmiles format describing the molecule. The string needs
            to have at least one coarse and one fine level. The fine level is
            used to match against the existing molecule, so you need to make
            sure that the labels in the CGsmiles string match the molecule
            atomnames or elements depending on if the string is at all-atom level.
        all_atom: bool
            default False; is the fine resolution an all-atom molecule
        topology: polyply.src.topology.Topology
            a topology object in case one needs to guess the masses
        """
        # we need to guess elements
        math_on = 'atomname'
        if all_atom:
            match_on = 'element'           
            self.set_element_from_mass(topology)

        new_meta_mol = self.from_cgsmiles_str(self.force_field,
                                              cgsmiles_str,
                                              self.mol_name,
                                              seq_only=False,
                                              all_atom=all_atom)
        def _node_match(n1, n2):
            if all_atom:
                return n1[match_on] == n2[match_on]
        mapping = find_one_ismags_match(new_meta_mol.molecule, self.molecule, node_match=_node_match)
        res_mapping = {to_node: new_meta_mol.molecule.nodes[from_node]['resname'] for from_node, to_node in mapping.items()}
        self.relabel_and_redo_res_graph(res_mapping)

    @property
    def search_tree(self):

        if self.__search_tree is None:
            if self.root is None:
                self.root =_find_starting_node(self)
            if self.dfs:
                self.__search_tree = nx.bfs_tree(self, source=self.root)
            else:
                self.__search_tree = nx.dfs_tree(self, source=self.root)

        return self.__search_tree

    @staticmethod
    def _block_graph_to_res_graph(block):
        """
        Generate a residue graph from the nodes of `block`.

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
        Constructs a MetaMolecule from a list of monomers representing
        a linear molecule.

        Parameters
        ----------
        force_field: :class:`vermouth.forcefield.ForceField`
            the force-field that must contain the block
        monomers: list[:class:`polyply.meta_molecule.Monomer`]
            a list of Monomer tuples
        mol_name: str
            name of the molecule

        Returns
        -------
        :class:`polyply.MetaMolecule`
        """

        meta_mol_graph = cls(force_field=force_field, mol_name=mol_name)
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
    def from_sequence_file(cls, force_field, file_path, mol_name):
        """
        Generate a meta_molecule from known sequence file parsers.
        For an up-to-date list of file-parsers see the
        MetaMolecule.parsers class variable.a

        Parameters
        ----------
        force_field: :class:`vermouth.forcefield.ForceField`
        file_path: :class:`pathlib.Path`
            the path to the file
        mol_name: str
            name of the meta-molecule

        Returns
        -------
        :class:`polyply.MetaMolecule`

        Raises
        ------
        IOError
            if the file format is unkown.
        """
        extension = file_path.suffix.casefold()[1:]
        if extension in cls.parsers:
            graph = cls.parsers[extension](file_path)
        else:
            msg = f"File format {extension} is unkown."
            raise IOError(msg)

        meta_mol = cls(graph, force_field=force_field, mol_name=mol_name)
        return meta_mol

    @classmethod
    def from_itp(cls, force_field, itp_file, mol_name):
        """
        Constructs a :class::`MetaMolecule` from an itp file.
        This will automatically set the MetaMolecule.molecule
        attribute.

        Parameters
        ----------
        force_field: :class:`vermouth.forcefield.ForceField`
        itp_file: str or :class:`pathlib.Path`
            the path to the file
        mol_name: str
            name of the meta-molecule

        Returns
        -------
        :class:`polyply.MetaMolecule`
        """
        with open(itp_file) as file_:
            lines = file_.readlines()
            read_itp(lines, force_field)

        _make_edges(force_field)

        graph = MetaMolecule._block_graph_to_res_graph(force_field.blocks[mol_name])
        meta_mol = cls(graph, force_field=force_field, mol_name=mol_name)
        meta_mol.molecule = force_field.blocks[mol_name].to_molecule()
        return meta_mol

    #TODO Add from_molecule method
    @classmethod
    def from_block(cls, force_field, mol_name):
        """
        Constructs a :class::`MetaMolecule` from an vermouth.molecule.
        The force-field must contain the block with mol_name from
        which to create the MetaMolecule. This function automatically
        sets the MetaMolecule.molecule attribute.

        Parameters
        ----------
        force_field: :class:`vermouth.forcefield.ForceField`
            the force-field that must contain the block
        mol_name: str
            name of the block matching a key in ForceField.blocks

        Returns
        -------
        :class:`polyply.MetaMolecule`
        """
        _make_edges(force_field)
        block = force_field.blocks[mol_name]
        graph = MetaMolecule._block_graph_to_res_graph(block)
        meta_mol = cls(graph, force_field=force_field, mol_name=mol_name)
        meta_mol.molecule = force_field.blocks[mol_name].to_molecule()
        return meta_mol

    @classmethod
    def from_cgsmiles_str(cls,
                          force_field,
                          cgsmiles_str,
                          mol_name,
                          seq_only=True,
                          all_atom=False):
        """
        Constructs a :class::`MetaMolecule` from an CGSmiles string.
        The force-field must contain the block with mol_name from
        which to create the MetaMolecule. This function automatically
        sets the MetaMolecule.molecule attribute.

        Parameters
        ----------
        force_field: :class:`vermouth.forcefield.ForceField`
            the force-field that must contain the block
        cgsmiles_str:
            the CGSmiles string describing the molecule graph
        mol_name: str
            name of the block matching a key in ForceField.blocks
        seq_only: bool
            if the string only describes the sequence; if this is False
            then the molecule attribute is set
        all_atom: bool
            if the last molecule in the sequence is at all-atom resolution
            can only be used if seq_only is False

        Returns
        -------
        :class:`polyply.MetaMolecule`
        """
        if seq_only and all_atom:
            msg = "You cannot define a sequence at all-atom level.\n"
            raise IOError(msg)

        # check if we have multiple resolutions
        if cgsmiles_str.count('{') == 1:
            meta_graph = read_cgsmiles(cgsmiles_str)
            take_resname_from = 'fragname'
        elif seq_only:
            # initalize the cgsmiles molecule resolver
            resolver = MoleculeResolver.from_string(cgsmiles_str,
                                                    last_all_atom=all_atom)
            # grab the last graph of the resolve iter
            _, meta_graph = resolver.resolve_all()
            take_resname_from = 'atomname'
        else:
            # initalize the cgsmiles molecule resolver
            take_resname_from = 'fragname'
            resolver = MoleculeResolver.from_string(cgsmiles_str,
                                                    last_all_atom=all_atom)
            meta_graph, molecule = resolver.resolve_all()

        # we have to set some node attribute accoding to polyply specs
        for node in meta_graph.nodes:
            resname = meta_graph.nodes[node][take_resname_from]
            meta_graph.nodes[node]['resname'] = resname
            if not seq_only:
                for atom in meta_graph.nodes[node]['graph'].nodes:
                    meta_graph.nodes[node]['graph'].nodes[atom]['resname'] = resname
                    meta_graph.nodes[node]['graph'].nodes[atom]['resid'] = node + 1
                    molecule.nodes[atom]['resname'] = resname
                    molecule.nodes[atom]['resid'] = node + 1

            if 'atomname' in meta_graph.nodes[node]:
                del meta_graph.nodes[node]['atomname']
            meta_graph.nodes[node]['resid'] = node + 1

        meta_mol = cls(meta_graph, force_field=force_field, mol_name=mol_name)
        if not seq_only:
            meta_mol.molecule = molecule

        return meta_mol
