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
"""
Test the fragment finder for itp_to_ff.
"""
import random
import pytest
import networkx as nx
from vermouth.forcefield import ForceField
import polyply

@pytest.mark.parametrize(
    "match_keys, node1, node2, expected",
    [
        (["element"], {"element": "C"}, {"element": "C"}, True),
        (["element"], {"element": "H"}, {"element": "O"}, False),
        (["element", "charge"], {"element": "N", "charge": 0}, {"element": "N", "charge": 1}, False),
        (["element", "charge"], {"element": "O", "charge": -1}, {"element": "O", "charge": -1}, True),
    ],
)
def test_node_match(match_keys, node1, node2, expected):
    # molecule and terminal label don't matter
    frag_finder = polyply.src.fragment_finder.FragmentFinder(None)
    frag_finder.match_keys = match_keys
    assert frag_finder._node_match(node1, node2) == expected

def _scramble_nodes(graph):
    element_to_masses = {"O": 16,
                         "N": 14,
                         "C": 12,
                         "S": 32,
                         "H": 1}
    # Get a list of all nodes in the original graph
    nodes = list(graph.nodes())
    # Generate a randomized list of new node names/indices
    randomized_nodes = nodes.copy()
    random.shuffle(randomized_nodes)
    # Create a mapping from old nodes to new nodes
    node_mapping = {old_node: new_node for old_node, new_node in zip(nodes, randomized_nodes)}
    # Generate a new graph by applying the mapping to the original graph
    randomized_graph = nx.relabel_nodes(graph, node_mapping)
    for node in randomized_graph.nodes:
        for attr in ['resid', 'resname']:
            del randomized_graph.nodes[node][attr]
        ele = randomized_graph.nodes[node]['element']
        randomized_graph.nodes[node]['mass'] = element_to_masses[ele]
    return randomized_graph

@pytest.mark.parametrize(
    "big_smile, resnames",
    [
     # two residues no branches
     ("{[#CH3][#PEO]|4[#CH3]}.{#PEO=[$]COC[$],#CH3=[$]C}",
      ["CH3", "PEO"],
     ),
     # three residues no branches
     ("{[#OH][#PEO]|4[#CH3]}.{#PEO=[$]COC[$],#CH3=[$]C,#OH=[$]O}",
      ["CH3", "PEO", "OH"],
     ),
     # simple branch expansion
    ("{[#PMA]([#PEO][#PEO][#OH])|3}.{#PEO=[$]COC[$],#PMA=[>]CC[<]C(=O)OC[$],#OH=[$]O}",
    ["PMA", "PEO", "OH"]),
    # something with sulphur
    ("{[#P3HT]|3}.{#P3HT=CCCCCCC1=C[$]SC[$]=C1}",
    ["P3HT"])
    ])
def test_extract_fragments(big_smile, resnames):
    ff = ForceField("new")
    meta = polyply.MetaMolecule.from_cgsmiles_str(force_field=ff,
                                                  cgsmiles_str=big_smile,
                                                  mol_name='ref',
                                                  seq_only=False,
                                                  all_atom=True)
    ref_fragments = {}
    for meta_node in meta.nodes:
        fragname = meta.nodes[meta_node]["fragname"]
        if fragname not in ref_fragments:
            ref_fragments[fragname] = meta.nodes[meta_node]["graph"]
            nx.set_node_attributes(ref_fragments[fragname], fragname, "resname")

    # strips resid, resname, and scrambles order
    target_molecule = _scramble_nodes(meta.molecule)

    # initialize the fragment finder
    frag_finder = polyply.src.fragment_finder.FragmentFinder(target_molecule)
    fragments, res_graph = frag_finder.extract_unique_fragments(meta.molecule)

    def _res_node_match(a, b):
        return a['resname'] == b['resname']

    def _frag_node_match(a, b):
        for attr in ['element', 'resname']:
            if a[attr] != b[attr]:
                return False
        return True

    assert set(fragments.keys()) == set(resnames)
    print(meta.nodes(data=True))
    print(res_graph.nodes(data=True))
    assert nx.is_isomorphic(res_graph, meta, node_match=_res_node_match)
#    for resname in resnames:
#        assert nx.is_isomorphic(fragments[resname],
#                                ref_fragments.blocks[resname],
#                                node_match=_frag_node_match)
