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

    Returns:
    --------
    bool
    """
    return node1["element"] == node2["element"]

class FragmentFinder():
    """
    Find, label and extract unique fragments from a vermouth.molecule.Molecule.

    Wrire process HERE
    """

    def __init__(self, molecule, prefix):
        """
        Initalize the fragment finder with a molecule, setting the
        resid attribute to None, and correctly assining elements
        based on atomic masses.

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
        """
        self.max_by_resid = {}
        self.ter_prefix = prefix
        self.resid = 1
        self.res_assigment = []
        self.assigned_atoms = []
        self.molecule = molecule
        self.known_atom = None
        self.match_keys = ['element', 'mass'] #, 'charge']
        self.masses_to_element = {16: "O",
                                  12: "C",
                                  32: "S",
                                   1: "H"}

        # resids are not reliable so we set them all to None
        nx.set_node_attributes(self.molecule, None, "resid")

        # set the element attribute for each atom in the
        # molecule
        for node in self.molecule.nodes:
            mass = round(self.molecule.nodes[node]["mass"])
            self.molecule.nodes[node]["element"] = self.masses_to_element[mass]

    def _node_match(self, node1, node2):
        for attr in self.match_keys:
            if node1[attr] != node2[attr]:
                return False
        return True

    def label_fragment_from_graph(self, fragment_graph):
        """
        For the `self.molecule` label all atoms that match
        the `fragment_graph` with a resid attribute and set
        the atom-name to the element name plus index relative
        to the atoms in the fragment.

        Parameters
        ----------
        fragment_graph: nx.Graph
            graph describing the fragment; must have the
            element attribute
        """
        # find all isomorphic matches to the target fragments
        GM = nx.isomorphism.GraphMatcher(self.molecule,
                                         fragment_graph,
                                         node_match=_element_match,
                                        )
        template_atoms = list(fragment_graph.nodes)
        # the below statement scales super duper extra poorly
        resname = list(nx.get_node_attributes(fragment_graph, "resname").values())[0]
        raw_matchs = GM.subgraph_isomorphisms_iter()
        # loop over all matchs and check if the atoms are already
        # assigned - symmetric matches must be skipped
        for current_match in raw_matchs:
            # the graph matcher can return the matchs in any order so we need to sort them
            # according to our tempalte molecule
            rev_current_match = {val: key for key, val in current_match.items()}
            atoms = [ rev_current_match[template_atom] for template_atom in template_atoms]
            if frozenset(atoms) not in self.res_assigment and not any([atom in self.assigned_atoms for atom in atoms]):
                self.res_assigment.append(frozenset(atoms))
                for idx, atom in enumerate(atoms):
                    self.molecule.nodes[atom]["resid"] = self.resid
                    self.molecule.nodes[atom]["atomname"] = self.molecule.nodes[atom]["element"] + str(idx)
                    self.molecule.nodes[atom]["resname"] = resname
                    self.max_by_resid[self.resid] = idx
                    self.known_atom = atom
                    self.assigned_atoms.append(atom)
                    print(self.molecule.nodes[atom]["element"])
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
        # now we make the residue graph and find all unique residues
        unique_fragments = {}
        res_graph = make_residue_graph(self.molecule)
        had_resnames = {}
        for node in res_graph.nodes:
            resname = res_graph.nodes[node]['resname']
            # this fragment is terminal located so we give it a special prefix
            fragment = res_graph.nodes[node]['graph']
            if res_graph.degree(node) == 1:
               resname = resname + self.ter_prefix
               nx.set_node_attributes(self.molecule, {node: resname for node in fragment.nodes} ,"resname")
            # here we extract the fragments and set appropiate residue names
            for other_frag in unique_fragments.values():
                if nx.is_isomorphic(fragment, other_frag, node_match=self._node_match):
                    # it can happen that two fragments are completely isomorphic but have different
                    # atom names because we don't know the order of atoms when looping over the molecule
                    # and setting the names. In this case we simply take the atom-names of the known
                    # fragment. Better ideas anyone?
                    mapping = find_one_ismags_match(fragment, other_frag, self._node_match)
                    if mapping:
                        for source, target in mapping.items():
                            self.molecule.nodes[target]['atomname'] = self.molecule.nodes[source]['atomname']
                        break
            else:
                if resname in unique_fragments:
                    resname = resname + "_" + str(had_resnames[resname] + 1)
                    nx.set_node_attributes(self.molecule, {node: resname for node in fragment.nodes} ,"resname")
                else:
                    had_resnames[resname] = 0
                unique_fragments[resname] = fragment

        return unique_fragments
