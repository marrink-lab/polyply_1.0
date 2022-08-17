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
import json
import networkx as nx
from networkx.readwrite import json_graph
import pytest
from polyply import gen_seq, TEST_DATA
from polyply.src.gen_seq import (_add_edges, _branched_graph, _tag_nodes,
                                 MacroString, generate_seq_graph,
                                 _find_terminal_nodes, _apply_termini_modifications)


def test_add_edge():
    graph = nx.Graph()
    graph.add_nodes_from(list(range(0, 6)))
    graph.add_edges_from([(0, 1), (1, 2),
                          (3, 4), (4, 5)])
    nx.set_node_attributes(graph,
                           {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2},
                           "seqid")
    edges = "1-2"
    _add_edges(graph, edges, 1, 2)
    assert graph.has_edge(1, 5)


@pytest.mark.parametrize("branching_f, n_levels, ref_edges", (
    (2, 3, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
    (1, 4, [(0, 1), (1, 2), (2, 3)])
))
def test_branched_graph(branching_f, n_levels, ref_edges):
    graph = _branched_graph("test", branching_f, n_levels)
    resnames = {idx: "test" for idx in graph.nodes}
    assert len(graph.edges) == len(ref_edges)
    for edge in ref_edges:
        graph.has_edge(edge[0], edge[1])
    assert nx.get_node_attributes(graph, "resname") == resnames


@pytest.mark.parametrize("macro_str, ref_edges, seed, resnames", (
    ("A:3:1:PEO-1.", [(0, 1), (1, 2)], None,
     {0: "PEO", 1: "PEO", 2: "PEO"}),
    ("A:3:2:PPI-1.",
     [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
     None,
     {0:"PPI", 1:"PPI", 2:"PPI", 3:"PPI", 4:"PPI", 5:"PPI", 6:"PPI"}),
    ("A:4:1:PEO-0.5,PPO-0.5",
     [(0, 1), (1, 2), (2, 3)],
     69034,
     {0:"PPO", 1:"PPO", 2:"PPO", 3:"PEO"}),
    ("A:4:1:PEO-0.3,PPO-0.2",
     [(0, 1), (1, 2), (2, 3)],
     69034,
     {0:"PPO", 1:"PPO", 2:"PPO", 3:"PEO"})
))
def test_interpret_macro_string(macro_str, ref_edges, seed, resnames):
    ref_graph = nx.Graph()
    ref_graph.add_edges_from(ref_edges)
    macro = MacroString(macro_str)
    macro_graph = macro.gen_graph(seed=seed)
    resnames_macro = nx.get_node_attributes(macro_graph, "resname")
    assert resnames == resnames_macro
    assert ref_graph.edges == macro_graph.edges


def test_generate_seq_graph():
    macro_str_A = "A:6:1:A-1."
    macro_str_B = "B:2:4:B-1."

    macros = {"A": MacroString(macro_str_A),
              "B": MacroString(macro_str_B)}

    seq_graph = generate_seq_graph(
        ["A", "B", "A"], macros, ["0:1:2-0", "1:2:1-1"])
    graphA = macros["A"].gen_graph()
    graphB = macros["B"].gen_graph()

    ref_graph = nx.disjoint_union(graphA, graphB)
    ref_graph = nx.disjoint_union(ref_graph, graphA)
    ref_graph.add_edges_from([(2, 6), (7, 12)])
    assert nx.is_isomorphic(ref_graph, seq_graph)


@pytest.mark.parametrize("edge_str",
                         ("0:1:2-100",
                          "0:100:2-0"))
def test_generate_seq_edge_error(edge_str):
    macro_strA = "A:6:1:A-1."
    macro_strB = "B:2:4:B-1."

    macros = {"A": MacroString(macro_strA),
              "B": MacroString(macro_strB)}

    with pytest.raises(IOError):
        generate_seq_graph(["A", "B"], macros, [edge_str])


def test_find_termini():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])
    termini = _find_terminal_nodes(graph)
    for node in [0, 2, 4, 5]:
        assert node in termini


def test_annote_modifications():
    modf = ["1:NH2", "2:NH3"]
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])
    nx.set_node_attributes(graph,
                           {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2},
                           "seqid"
                          )
    nx.set_node_attributes(graph,
                           {0: "PPI", 1: "PPI", 2: "PPI", 3: "PPI", 4: "PPI", 5: "PPI"},
                           "resname"
                          )
    _apply_termini_modifications(graph, modf)
    assert nx.get_node_attributes(graph, "resname") == {0: "NH2", 1: "PPI",
                                                        2: "NH2", 3: "PPI",
                                                        4: "NH3", 5: "NH3"}

#<seqID:tag_word:value1-probability,value2-probability>
@pytest.mark.parametrize('tags, expected, seed',(
                         # only one sequence, one value, prob=1
                        [["1:chiral:R-1."],
                         {0: "R", 1: "R", 2: "R"},
                         None
                        ],
                         # two tags different seqID value prob=1
                        [["1:chiral:R-1.", "2:chiral:S-1."],
                         {0: "R", 1: "R", 2: "R", 3: "S", 4: "S", 5: "S"},
                         None
                        ],
                        # randomly replace nodes in one sequence
                        [["1:chiral:R-0.5,S-0.5", "2:chiral:Q-1."],
                         {0: "S", 1: "R", 2: "S", 3: "Q", 4: "Q", 5: "Q"},
                         23303
                        ],
                        # randomly replace nodes in both sequences
                        [["1:chiral:R-0.5,S-0.5", "2:chiral:P-0.5,D-0.5"],
                         {0: "S", 1: "R", 2: "S", 3: "D", 4: "P", 5: "D"},
                         23303
                        ]
                        ))
def test_tag_nodes(tags, expected, seed):
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])
    nx.set_node_attributes(graph,
                           {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2},
                           "seqid"
                          )
    nx.set_node_attributes(graph,
                           {0: "PPI", 1: "PPI", 2: "PPI", 3: "PPI", 4: "PPI", 5: "PPI"},
                           "resname"
                          )
    _tag_nodes(graph, tags, seed)
    assert nx.get_node_attributes(graph, "chiral") == expected

@pytest.mark.parametrize('args, ref_file', (
    (dict(outpath=TEST_DATA + "/gen_seq/output/PPI.json",
          macro_strings=["A:3:2:N1-1.0"],
          seq=["A", "A"],
          name="test",
          connects=["0:1:0-0"]),
     TEST_DATA + "/gen_seq/ref/PPI_ref.json"),
    (dict(outpath=TEST_DATA + "/gen_seq/output/PEO_PS.json",
          macro_strings=["A:11:1:PEO-1", "B:11:1:PS-1"],
          connects=["0:1:10-0"],
          name="test",
          seq=["A", "B"]),
     TEST_DATA + "/gen_seq/ref/PEO_PS_ref.json"),
    (dict(outpath=TEST_DATA + "/gen_seq/output/lysoPEG.json",
          inpath=[TEST_DATA + "/gen_seq/input/molecule_0.itp"],
          macro_strings=["A:5:1:PEG-1.0"],
          from_file=["PROT:molecule_0"],
          name="test",
          seq=["PROT", "A"],
          connects=["0:1:0-0"]),
     TEST_DATA + "/gen_seq/ref/lyso_PEG.json")
))
def test_gen_seq(args, ref_file):
    gen_seq(**args)

    with open(ref_file) as _file:
        js_graph = json.load(_file)
        ref_graph = json_graph.node_link_graph(js_graph)

    with open(args["outpath"]) as _file:
        js_graph = json.load(_file)
        out_graph = json_graph.node_link_graph(js_graph)

    assert nx.is_isomorphic(out_graph, ref_graph)
    assert nx.get_node_attributes(
        out_graph, "resname") == nx.get_node_attributes(ref_graph, "resname")
