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
from collections import defaultdict
import networkx as nx
from vermouth.graph_utils import make_residue_graph
from polyply.src.graph_utils import find_one_graph_match

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
    Label the atoms of an all-atom target molecule (resid, resname,
    atomname) by aligning it with an already-labelled all-atom
    reference molecule graph.

    The target molecule and the reference graph are aligned using a
    graph isomorphism match (see `_match_reference_to_molecule`). For
    performance, this match is computed on reduced copies of both
    graphs from which hydrogen and virtual-site atoms have been
    removed (`remove_special_nodes`), matching only on the atom
    properties listed in `self.match_keys` (by default just
    `element`). The resulting mapping is used to copy resid, resname,
    and atomname from the reference graph onto the matched nodes of
    the target molecule (`_label_matched_atoms`).

    Because hydrogen and virtual-site atoms are excluded from the
    match, they are not labelled by the isomorphism itself. Since
    every such atom must belong to exactly one residue, they are
    labelled afterwards by inspecting the anchor atom(s) they are
    bonded to or constructed from: a hydrogen or virtual atom is
    assigned the resid and resname of its already-labelled neighbor(s)
    (`_label_unmatched_atoms`). Terminal hydrogen atoms that form
    their own residue in the reference graph (e.g. a capping -H) have
    no direct counterpart among the matched atoms, so they are
    resolved separately beforehand via their anchor atom
    (`_label_hydrogen_termini`).

    Finally, a residue graph is built from the fully labelled target
    molecule, and for each resname the most central residue instance
    (highest betweenness centrality, so as to avoid picking terminal
    residues) is kept as the representative fragment
    (`_select_unique_fragments`).

    This class makes a number of implicit assumptions:

    - the molecule is connected and acyclic
    - the residue graph of the molecule is linear
    - the nodes by index increase with increasing resid order
    """

    def __init__(self, molecule):
        """
        Initalize the fragment finder with a molecule, setting the
        resid attribute to None, and correctly assining elements
        based on atomic masses.

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
            the molecule to match against

        Attributes
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
            the molecule to match against
        match_keys: `list[str]`
            molecule properties to use in matching the fragment
            graphs in the second stage.
        masses_to_element: dict[int][str]
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

    def _match_reference_to_molecule(self, reference_graph):
        """
        Find a subgraph isomorphism between the target molecule and
        the reference graph. Hydrogen and virtual-site atoms are
        excluded from both graphs before matching, since they are
        not needed to uniquely determine the mapping and excluding
        them is cheaper.

        Parameters
        ----------
        reference_graph: :class:`networkx.Graph`

        Returns
        -------
        dict
            mapping of target molecule nodes to reference_graph nodes
        """
        match_target = remove_special_nodes(self.molecule)
        match_reference = remove_special_nodes(reference_graph)
        return find_one_graph_match(match_target,
                                    match_reference,
                                    node_match=self._node_match)

    def _label_matched_atoms(self, mapping, reference_graph):
        """
        Copy resname, resid, and atomname from the reference graph
        onto every target molecule node covered by `mapping`.

        Parameters
        ----------
        mapping: dict
            target molecule nodes mapped to reference_graph nodes,
            as returned by `_match_reference_to_molecule`.
        reference_graph: :class:`networkx.Graph`
        """
        for target, ref in mapping.items():
            for attr in ['resname', 'resid', 'atomname']:
                self.molecule.nodes[target][attr] = reference_graph.nodes[ref][attr]

    def _label_hydrogen_termini(self, mapping, reference_graph):
        """
        Label terminal hydrogen atoms that form their own residue in
        the reference graph (e.g. a capping -H). Such atoms have no
        counterpart in `mapping`, because hydrogens are excluded from
        the isomorphism match. Instead, for each hydrogen terminal
        residue in the reference graph, the corresponding anchor atom
        is looked up in the target molecule and the one still
        unlabelled hydrogen neighbor is assigned the terminal's
        resname, resid, and atomname.

        Parameters
        ----------
        mapping: dict
            target molecule nodes mapped to reference_graph nodes,
            as returned by `_match_reference_to_molecule`.
        reference_graph: :class:`networkx.Graph`
        """
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

    def _label_unmatched_atoms(self):
        """
        Label all remaining target molecule nodes that were excluded
        from the isomorphism match and not already handled by
        `_label_hydrogen_termini` (i.e. hydrogen atoms and virtual
        sites). Every such atom must belong to exactly one residue,
        so it is assigned the resid and resname of its already
        labelled neighbor(s) (its anchor atoms). Atoms sharing the
        same set of anchor atomnames get systematically generated,
        unique atomnames.
        """
        _names = {}
        _counter = {}
        for node in self.molecule.nodes:
            if self.molecule.nodes[node].get('resid', False):
                continue
            element = self.molecule.nodes[node].get('element', None)
            anchors = [anchor for anchor in self.molecule.neighbors(node)
                       if self.molecule.nodes[anchor].get('resid', False)]
            anchors_names = tuple(self.molecule.nodes[anchor]["atomname"] for anchor in anchors)
            resids = [self.molecule.nodes[anchor]["resid"] for anchor in anchors]
            assert len(set(resids)) == 1
            self.molecule.nodes[node]["resid"] = resids[0]
            self.molecule.nodes[node]["resname"] = self.molecule.nodes[anchors[0]]["resname"]
            if anchors_names in _names:
                atomname = _names[anchors_names]
            else:
                atomname = element[0] + f"{len(_names)}"
                _names[anchors_names] = atomname
            idx = _counter.get((anchors_names, resids[0]), 0)
            _counter[(anchors_names, resids[0])] = idx + 1
            self.molecule.nodes[node]["atomname"] = atomname + f"{idx}"

    def _check_fragment_consistency(self):
        """
        Verify that all residues sharing a resname consist of the same atoms.

        Only one block per resname is written to the force field (see
        `_select_unique_fragments`), so a resname whose instances differ in
        composition cannot be represented. That happens when a bonding
        operator of a fragment is used in some places but left open in
        others, because an open operator is capped with a hydrogen. The
        resulting force field would be silently wrong, so we refuse it here
        rather than let it fail later when the parameters are applied.

        Raises
        ------
        IOError
            if any resname occurs with more than one set of atomnames
        """
        compositions = defaultdict(lambda: defaultdict(list))
        for res in self.res_graph:
            attrs = self.res_graph.nodes[res]
            graph = attrs['graph']
            signature = tuple(sorted(graph.nodes[node]['atomname'] for node in graph))
            compositions[attrs['resname']][signature].append(attrs['resid'])

        problems = []
        for resname, variants in sorted(compositions.items()):
            if len(variants) == 1:
                continue
            common = set.intersection(*(set(sig) for sig in variants))
            problems.append(f"Residue '{resname}' occurs with {len(variants)} "
                            "different sets of atoms:")
            for signature, resids in sorted(variants.items(), key=lambda item: len(item[0])):
                extra = sorted(set(signature) - common)
                shown = ', '.join(str(resid) for resid in sorted(resids)[:5])
                if len(resids) > 5:
                    shown += ', ...'
                problems.append(f"  {len(signature)} atoms (resid {shown})"
                                + (f"; extra atoms: {' '.join(extra)}" if extra else ""))
        if problems:
            raise IOError(
                "\n".join(problems)
                + "\nIn your CGsmiles string you describe two residues that are not equivalent with "
                  "the same residue name. However, only one block per residue name is written to the "
                  " force field, so these cannot all be described. Usually a bonding operator of the "
                  " residue is used in some places and left open in others, where it is capped with "
                  " a hydrogen. Make that cap explicit in the CGsmiles string by adding a terminal "
                  "residue (e.g. #Hter=[>1][<1][H]) everywhere the operator is unused. Note that the "
                  "name of such a residue must contain 'ter'.")

    def _select_unique_fragments(self):
        """
        Collect one representative residue graph per resname from
        `self.res_graph`, preferring the most central residue
        instance (highest betweenness centrality) so as to avoid
        picking terminal residues.

        Returns
        -------
        dict[str, nx.Graph]
            resname mapped to a representative fragment graph
        """
        unique_fragments = {}
        frag_centrality = {}
        centrality = nx.betweenness_centrality(self.res_graph)
        for res in self.res_graph:
            resname = self.res_graph.nodes[res]['resname']
            if resname not in unique_fragments or frag_centrality[resname] < centrality[res]:
                unique_fragments[resname] = self.res_graph.nodes[res]['graph']
                frag_centrality[resname] = centrality[res]
        return unique_fragments

    def extract_unique_fragments(self, reference_graph):
        """
        Label the target molecule according to `reference_graph` and
        extract one representative fragment graph per residue type.

        Parameters
        ----------
        reference_graph: :class:`networkx.Graph`
            an all-atom reference graph already annotated with
            resname, resid, and atomname on every node.

        Returns
        -------
        dict[str, nx.Graph], :class:`networkx.Graph`
            resname mapped to a representative fragment graph, and
            the full residue graph of the labelled target molecule
        """
        mapping = self._match_reference_to_molecule(reference_graph)
        self._label_matched_atoms(mapping, reference_graph)
        # potentially hydrogen atoms may be their own residues;
        # we deal with those first before the generic anchor-based
        # labelling of unmatched atoms
        self._label_hydrogen_termini(mapping, reference_graph)
        self._label_unmatched_atoms()

        self.make_res_graph()
        self._check_fragment_consistency()
        unique_fragments = self._select_unique_fragments()
        return unique_fragments, self.res_graph
