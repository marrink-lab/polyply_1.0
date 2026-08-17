# Copyright 2024 Dr. Fabian Gruenewald
#
# Licensed under the PolyForm Noncommercial License 1.0.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://polyformproject.org/licenses/noncommercial/1.0.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import networkx as nx
from vermouth.graph_utils import make_residue_graph
from polyply.src.graph_utils import find_one_ismags_match

def remove_special_nodes(graph, elements=["virtual", "H"]):
    """
    Returns a subgraph of a molecule sans atoms that are
    specified in the `elements` list and annotates the
    number of removed atoms on each remaining node.
    """
    not_hnodes = [node for node in graph.nodes if graph.nodes[node]["element"] not in elements]
    not_h_graph = graph.subgraph(not_hnodes)
    for node in not_hnodes:
        for element in elements:
            count = sum(1 for neighbor in nx.neighbors(graph, node) if graph.nodes[neighbor]["element"] == element)
            tag = f"{element}count"
            not_h_graph.nodes[node][tag] = count
    return not_h_graph

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

    def __init__(self, molecule):
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
        self.molecule = molecule
        # we need to make a copy and remove the hydrogen atoms
        # and virtual sides
        #self.molecule = molecule.copy()
        self.match_keys = ['element'] #, 'mass', 'degree'] #, 'charge']
        self.masses_to_element = {16: "O",
                                  14: "N",
                                  12: "C",
                                  19: "F",
                                  35: "Cl",
                                  32: "S",
                                   1: "H"}
        self.res_graph = None

        if self.molecule:
            # resids are not reliable so we set them all to None
            nx.set_node_attributes(self.molecule, None, "resid")

            # set the element attribute for each atom in the
            # molecule
            vs_nodes = []
            for node in self.molecule.nodes:
                # we also need to filter out virtual-sides
                mass = round(self.molecule.nodes[node]["mass"])
                if mass == 0:
                    self.molecule.nodes[node]["element"] = "virtual"
                else:
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

    def extract_unique_fragments(self, reference_graph):
        """
        Call the label_fragment method for multiple fragments.

        Parameters
        ----------
        fragment_graphs: list[nx.Graph]
        """
        # return the subgraph on which to match
        # that is sans hatoms and virtual nodes
        match_target = remove_special_nodes(self.molecule)
        match_reference = remove_special_nodes(reference_graph)
        # find one correspondance
        mapping = find_one_ismags_match(match_target,
                                        match_reference,
                                        node_match=self._node_match)

        # now assign the attributes from the reference graph to
        # the target molecule
        for target, ref in mapping.items():
            for neigh in self.molecule.neighbors(target):
                if self.molecule.nodes[neigh]["element"] != "H":
            for attr in ['resname', 'resid', 'atomname']:
                self.molecule.nodes[target][attr] = reference_graph.nodes[ref][attr]

        # potentially hydrogen atoms may be their own residues
        # we deal with those first
        rev_mapping = {value: key for key, value in mapping.items()}
        ref_resnames = nx.get_node_attributes(reference_graph, "resname")
        for node, resname in ref_resnames.items():
            if "ter" in resname and reference_graph.nodes[node]["element"] == "H":
                anchor = list(reference_graph.neighbors(node))[0]
                for target in self.molecule.neighbors(rev_mapping[anchor]):
                    if self.molecule.nodes[target]["element"] == "H":
                        break
                else:
                    raise IOError
                for attr in ['resname', 'resid', 'atomname']:
                    self.molecule.nodes[target][attr] = reference_graph.nodes[node][attr]
        # we are now left with some nodes that were not covered in the
        # mapping (e.g. hydrogen atoms or virtual atoms)
        _names = {}
        _counter = {}
        for node in self.molecule.nodes:
            node_name = self.molecule.nodes[node]["atomname"]
            if not self.molecule.nodes[node].get('resid', False):
                element = self.molecule.nodes[node].get('element', None)
                anchors = [ anchor for anchor in self.molecule.neighbors(node) if self.molecule.nodes[anchor].get('resid', False)]
                anchors_names = tuple(self.molecule.nodes[node]["atomname"] for node in anchors)
                resids = [self.molecule.nodes[anchor]["resid"] for anchor in anchors]
                assert len(set(resids)) == 1
                self.molecule.nodes[node]["resid"] = resids[0]
                self.molecule.nodes[node]["resname"] = self.molecule.nodes[anchors[0]]["resname"]
                if anchors_names in _names:
                    atomname = _names[anchors_names]
                else:
                   atomname = self.molecule.nodes[node]["element"][0] + f"{len(_names)}"
                   _names[anchors_names] = atomname
                idx = _counter.get((anchors_names, resids[0]), 0)
                _counter[(anchors_names, resids[0])] = idx + 1
                atomname = atomname + f"{idx}"
                self.molecule.nodes[node]["atomname"] = atomname

        # now we make the residue graph and extract
        self.make_res_graph()
        # finally we simply collect one graph per restype
        # which are the most central (i.e. avoid ends)
        unique_fragments = {}
        frag_centrality = {}
        centrality = nx.betweenness_centrality(self.res_graph)
        for res in self.res_graph:
            resname = self.res_graph.nodes[res]['resname']
            if resname not in unique_fragments or frag_centrality[resname] < centrality[res]:
                unique_fragments[resname] = self.res_graph.nodes[res]['graph']
                frag_centrality[resname] = centrality[res]
        return unique_fragments, self.res_graph
