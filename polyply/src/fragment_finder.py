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

import networkx as nx
from vermouth.graph_utils import make_residue_graph
from polyply.src.graph_utils import find_one_ismags_match

def _element_match(node1, node2):
    """
    Checks if the element attribute of two nodes
    is the same.

    Returns
    --------
    bool
    """
    return node1["element"] == node2["element"]

class FragmentFinder():
    """
    This class enables finding and labelling of fragments
    in the all-atom description of molecules. Fragments are
    small networkx graphs. It makes a number of implicit
    assumptions:

    - the molecule is connected and acyclic
    - the residue graph of the molecule is linear
    - the nodes by index increase with increasing resid order
    - the graphs provided as fragment graphs follow the sequence
      of residues. For example, given a polymer A5-B2-C3-A3
      residue sequence, fragments should be provided as a list
      A,B,C,A. The length of the block does not matter.

    The algorithm loops over the fragments and finds a match
    between a fragment and the molecule graph using a subgraph
    isomorphism based on the element attribute. This match is
    then used to set the degree attribute on the fragment. Next
    all other subgraph isomorphisms are found under the condition
    that each found match must connected to the previous residue.
    Nodes are labelled with a resid and resname. This part is done
    by the `self.label_fragment_from_graph` class method.

    Subsequently, the algorithm proceeds to merge all left-over
    atoms to the residue they are connected with assining a resid
    and resname from that residue. This procedure is done by
    `self.label_unmatched_atoms`.

    Finally, the code goes over all residues and assigns a prefix to
    all terminal residues. In addition residues with the same resname
    are compared to each other using a subgraph isomorphism and if
    they are not isomorphic as result of assigning left-over atoms,
    the resname is appended by a number.
    """

    def __init__(self, molecule, prefix):
        """
        Initalize the fragment finder with a molecule, setting the
        resid attribute to None, and correctly assining elements
        based on atomic masses.

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
        prefix: str
            the prefix used to label termini

        Attributes
        ----------
        max_by_resid: dict[int][int]
            number of atoms by resid
        ter_prefix: str
            the terminal prefix
        resid: int
            highest resid
        assigned_atoms: list[`abc.hashable`]
            atoms assinged to residues
        molecule: :class:`vermouth.molecule.Molecule`
            the molecule to match against
        known_atom: `abc.hashable`
            any atom that has been matched to a fragment
        match_keys: `list[str]`
            molecule properties to use in matching the fragment
            graphs in the second stage.
        masses_to_elements: dict[int][str]
            matches masses to elements
        res_graph: :class:`vermouth.molecule.Molecule`
            residue graph of the molecule
        """
        self.max_by_resid = {}
        self.ter_prefix = prefix
        self.resid = 1
        self.res_assigment = []
        self.assigned_atoms = []
        self.molecule = molecule
        self.known_atom = None
        self.match_keys = ['element', 'mass', 'degree'] #, 'charge']
        self.masses_to_element = {16: "O",
                                  14: "N",
                                  12: "C",
                                  32: "S",
                                   1: "H"}
        self.res_graph = None

        if self.molecule:
            # resids are not reliable so we set them all to None
            nx.set_node_attributes(self.molecule, None, "resid")

            # set the element attribute for each atom in the
            # molecule
            for node in self.molecule.nodes:
                mass = round(self.molecule.nodes[node]["mass"])
                self.molecule.nodes[node]["element"] = self.masses_to_element[mass]
                self.molecule.nodes[node]["degree"] = self.molecule.degree(node)

    def _node_match(self, node1, node2):
        """
        Check if two node dicts match.

        Parameters
        ----------
        node1: dict
        node2: dict

        Returns
        -------
        bool
        """
        for attr in self.match_keys:
            if node1[attr] != node2[attr]:
                return False
        return True

    # this could be a property??
    def make_res_graph(self):
        self.res_graph = make_residue_graph(self.molecule)

    def pre_match(self, fragment_graph):
        """
        Find one match of fragment graph in the molecule
        and then extract degrees and atom-types for further
        matching. This is a safety measure because even though
        the fragment graph is subgraph isomorphic the underlying
        itp parameters might not be.

        Parameters
        -----------
        fragment_graph: 'nx.Graph'
            must have attributes element for each node

        Returns
        -------
        'nx.Graph'
            the labelled fragment graph
        """
        template_atoms = list(fragment_graph.nodes)
        # find subgraph isomorphic matches to the target fragment
        # based on the element only
        GM = nx.isomorphism.GraphMatcher(self.molecule,
                                         fragment_graph,
                                         node_match=_element_match,)

        for one_match in GM.subgraph_isomorphisms_iter():
            rev_current_match = {val: key for key, val in one_match.items()}
            atoms = [ rev_current_match[template_atom] for template_atom in template_atoms]
            if self.is_valid_match(one_match, atoms)[0]:
                break

        for mol_atom, tempt_atom in one_match.items():
            for attr in self.match_keys:
                fragment_graph.nodes[tempt_atom][attr] = self.molecule.nodes[mol_atom][attr]
        return fragment_graph

    def is_valid_match(self, match, atoms):
        """
        Check if the found isomorphism match is valid.
        """
        # is the match connected to the previous residue
        if not self.is_connected_to_prev(match.keys(), self.assigned_atoms,):
            return False, 1
        # check if atoms are already assigned
        if frozenset(atoms) in self.res_assigment:
            return False, 2
        # check if there is any partial overlap
        if any([atom in self.assigned_atoms for atom in atoms]):
            return False, 3

        return True, 4

    def is_connected_to_prev(self, current, prev):
        """
        Check if the atoms in the lists current or
        prev are connected.

        Parameters
        ----------
        current: list[abc.hashable]
            list of current nodes
        prev: list[abc.hashable]
            list of prev nodes
        """
        # no atoms have been assigned
        if len(prev) == 0:
            return True

        for node in current:
            for neigh_node in self.molecule.neighbors(node):
                if neigh_node in prev:
                    return True
        return False

    def label_fragment_from_graph(self, fragment_graph):
        """
        For the `self.molecule` label all atoms, that match
        the `fragment_graph`, with a resid attribute and set
        the atom-name to the element name plus index relative
        to the atoms in the fragment.

        Parameters
        ----------
        fragment_graph: nx.Graph
            graph describing the fragment; must have the
            element attribute
        """
        # pre-match one residue and extract the atomtypes and degrees
        # this is needed to enforce symmetry in matching the other
        # residues
        fragment_graph = self.pre_match(fragment_graph)
        # find all isomorphic matches to the target fragments
        GM = nx.isomorphism.GraphMatcher(self.molecule,
                                         fragment_graph,
                                         node_match=self._node_match,
                                        )
        template_atoms = list(fragment_graph.nodes)
        resname = list(nx.get_node_attributes(fragment_graph, "resname").values())[0]
        raw_matchs = list(GM.subgraph_isomorphisms_iter())
        # loop over all matchs and check if the atoms are already
        # assigned - symmetric matches must be skipped
        for current_match in raw_matchs:
            # the graph matcher can return the matchs in any order so we need to sort them
            # according to our tempalte molecule
            rev_current_match = {val: key for key, val in current_match.items()}
            atoms = [ rev_current_match[template_atom] for template_atom in template_atoms]
            if self.is_valid_match(current_match, atoms)[0]:
                self.res_assigment.append(frozenset(atoms))
                for idx, atom in enumerate(atoms):
                    self.molecule.nodes[atom]["resid"] = self.resid
                    self.molecule.nodes[atom]["atomname"] = self.molecule.nodes[atom]["element"] + str(idx)
                    self.molecule.nodes[atom]["resname"] = resname
                    self.max_by_resid[self.resid] = idx
                    self.known_atom = atom
                    self.assigned_atoms.append(atom)
                self.resid += 1

    def label_fragments_from_graph(self, fragment_graphs):
        """
        Call the label_fragment method for multiple fragments.

        Parameters
        ----------
        fragment_graphs: list[nx.Graph]
        """
        for fragment_graph in fragment_graphs:
            self.label_fragment_from_graph(fragment_graph)

    def label_unmatched_atoms(self):
        """
        After all atoms have been assigned to target fragments using
        the label_fragment method all left-over atoms are assigned to
        the first fragment they are attached to. This method sets the
        atom-name to the element name and element count and resid
        attribute.
        """
        for from_node, to_node in nx.dfs_edges(self.molecule, source=self.known_atom):
            if not self.molecule.nodes[to_node]["resid"]:
                resid = self.molecule.nodes[from_node]["resid"]
                self.max_by_resid[resid] = self.max_by_resid[resid] + 1
                self.molecule.nodes[to_node]["resid"] = resid
                self.molecule.nodes[to_node]["resname"] = self.molecule.nodes[from_node]["resname"]
                self.molecule.nodes[to_node]["atomname"] = self.molecule.nodes[to_node]["element"] + str(self.max_by_resid[resid])

    def extract_unique_fragments(self, fragment_graphs):
        """
        Given a list of fragment-graphs assing all atoms to fragments and
        generate new fragments by assinging the left-over atoms to the
        connecting fragment. Fragments get a unique resid in the molecule.
        Then make the residue graph and filter out all unique residues
        and return them.

        Parameters
        ----------
        fragment_graphs: list[nx.Graph]

        Returns
        -------
        list[nx.Graph]
            all unique fragment graphs
        """
        # first we find and label all fragments in the molecule
        self.label_fragments_from_graph(fragment_graphs)
        # then we assign all left-over atoms to the existing residues
        self.label_unmatched_atoms()
        # make the residue graph
        self.make_res_graph()
        # now we make the residue graph and find all unique residues
        unique_fragments = {}
        had_resnames = {}
        for node in self.res_graph.nodes:
            resname = self.res_graph.nodes[node]['resname']
            # this fragment is terminal located so we give it a special prefix
            fragment = self.res_graph.nodes[node]['graph']
            if self.res_graph.degree(node) == 1:
               resname = resname + self.ter_prefix
               nx.set_node_attributes(self.molecule, {node: resname for node in fragment.nodes} ,"resname")
               nx.set_node_attributes(fragment, {node: resname for node in fragment.nodes} ,"resname")
            # here we extract the fragments and set appropiate residue names
            for other_frag in unique_fragments.values():
                if nx.is_isomorphic(fragment, other_frag, node_match=self._node_match):
                    mapping = find_one_ismags_match(fragment, other_frag, self._node_match)
                    if mapping:
                        for source, target in mapping.items():
                            self.molecule.nodes[target]['atomname'] = self.molecule.nodes[source]['atomname']
                        break
            else:
                if resname in unique_fragments:
                    resname = resname + "_" + str(had_resnames[resname] + 1)
                    nx.set_node_attributes(self.molecule, {node: resname for node in fragment.nodes} ,"resname")
                    nx.set_node_attributes(fragment, {node: resname for node in fragment.nodes} ,"resname")
                else:
                    had_resnames[resname] = 0
                unique_fragments[resname] = fragment

        # remake the residue graph since some resnames have changed
        self.make_res_graph()
        return unique_fragments
