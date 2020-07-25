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
import math
import networkx as nx
from networkx.readwrite import json_graph
import pytest
import argparse
import json
from pathlib import Path
import textwrap
from collections import Counter
from polyply import gen_seq, TEST_DATA
from polyply.src.gen_seq import _add_edges, _macro_to_graph, _random_macro_to_graph, interpret_macro_string, generate_seq_graph

def test_add_edge():
    graph = nx.Graph()
    graph.add_nodes_from(list(range(0, 6)))
    graph.add_edges_from([(0, 1), (1, 2),
                          (3, 4), (4, 5)])
    nx.set_node_attributes(graph, {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}, "seqid")
    edges = "1-2"
    _add_edges(graph, edges, 1, 2)
    assert graph.has_edge(1, 5)

@pytest.mark.parametrize("branching_f, n_levels, ref_edges",(
                        (2, 3, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
                        (1, 4, [(0, 1), (1, 2), (2, 3)])
                        ))
def test_macro_to_graph(branching_f, n_levels, ref_edges):
    graph = _macro_to_graph("test", branching_f, n_levels)
    resnames = {idx: "test" for idx in graph.nodes}
    assert len(graph.edges) == len(ref_edges)
    for edge in ref_edges:
        graph.has_edge(edge[0], edge[1])
    assert nx.get_node_attributes(graph, "resname") == resnames


@pytest.mark.parametrize("residues",(
                        'PEO-0.5,PPO-0.5',
                        'PEO-0.2,PPO-0.8',
                        'PEO-0.1,PPO-0.9'
                        ))
def test_random_macro_to_graph(residues):
    graph = _random_macro_to_graph(12, residues)
    resnames = nx.get_node_attributes(graph, "resname")
    total = len(graph.nodes)
    res_prob_A, res_prob_B = residues.split(",")
    res_A, prob_A = res_prob_A.split("-")
    res_B, prob_B = res_prob_B.split("-")
    resnames = Counter(nx.get_node_attributes(graph,"resname").values())
    assert math.isclose(resnames[res_A]/total, float(prob_A), abs_tol=0.07)
    assert math.isclose(resnames[res_B]/total, float(prob_B), abs_tol=0.07)


@pytest.mark.parametrize("macro_str, macro_type, ref_edges",(
                        ("A:PEO:3", "linear", [(0, 1), (1, 2)]),
                        ("A:PPI:2:3", "branched", [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
                        ("A:4:PEO-0.5,PPO-0.5","random-linear", [(0, 1), (1, 2), (2, 3)]),
                        ))
def test_interpret_macro_string(macro_str, macro_type, ref_edges):
    graph = nx.Graph()
    graph.add_edges_from(ref_edges)
    name, macro = interpret_macro_string(macro_str, macro_type, force_field=None)
    assert len(nx.get_node_attributes(macro, "resname")) == len(graph.nodes)
    assert graph.edges == macro.edges


def test_generate_seq_graph():
    graphA = nx.Graph()
    graphA.add_nodes_from(list(range(0, 6)))
    graphA.add_edges_from([(0, 1), (1, 2),
                          (3, 4), (4, 5)])
    graphB = nx.Graph()
    graphB.add_nodes_from(list(range(0, 5)))
    graphB.add_edges_from([(0, 1), (0, 2),
                          (0, 3), (0, 4)])

    macros = {"A":graphA, "B":graphB}

    seq_graph = generate_seq_graph(["A", "B", "A"], macros, ["0:1:2-0", "1:2:1-1"])
    ref_graph = nx.disjoint_union(graphA, graphB)
    ref_graph = nx.disjoint_union(ref_graph, graphA)
    ref_graph.add_edges_from([(2, 6), (7, 12)])
    assert nx.is_isomorphic(ref_graph, seq_graph)


class args:
     """
     Inpute Arguments for the sequence generator.
     """
     def __init__(self,
                  name=None,
                  ffpath=None,
                  outpath=None,
                  linear=None,
                  branched=None,
                  file_macros=None,
                  random_linear=None,
                  seq=None,
                  connects=None,
                  lib=None,
                  inpath=None):

                  self.name=name
                  self.lib=lib
                  self.ffpath=ffpath
                  self.outpath=outpath
                  self.linear=linear
                  self.branched=branched
                  self.from_file=file_macros
                  self.random_linear=random_linear
                  self.seq=seq
                  self.connects=connects
                  self.inpath=inpath


@pytest.mark.parametrize('_input, ref_file',(
                        (dict(outpath=TEST_DATA + "/gen_seq/output/PPI.json",
                          branched=["A:N1:2:3"],
                          seq=["A", "A"],
                          connects=["0:1:0-0"]),
                          TEST_DATA + "/gen_seq/ref/PPI_ref.json"),
                         (dict(outpath=TEST_DATA + "/gen_seq/output/PEO_PS.json",
                          linear=["A:PEO:11", "B:PS:11"],
                          connects=["0:1:10-0"],
                          seq=["A", "B"]),
                          TEST_DATA + "/gen_seq/ref/PEO_PS_ref.json"),
                         (dict(outpath=TEST_DATA + "/gen_seq/output/lysoPEG.json",
                          inpath=[TEST_DATA + "/gen_seq/input/molecule_0.itp"],
                          linear=["A:PEG:5"],
                          file_macros=["PROT:molecule_0"],
                          seq=["PROT", "A"],
                          connects=["0:1:0-0"]),
                          TEST_DATA + "/gen_seq/ref/lyso_PEG.json")
                          ))

def test_gen_seq(_input, ref_file):
    arguments = args(**_input)
    gen_seq(arguments)

    with open(ref_file) as _file:
         js_graph = json.load(_file)
         ref_graph = json_graph.node_link_graph(js_graph)

    with open(_input["outpath"]) as _file:
         js_graph = json.load(_file)
         out_graph = json_graph.node_link_graph(js_graph)

    assert nx.is_isomorphic(out_graph, ref_graph)
    assert nx.get_node_attributes(out_graph, "resname") == nx.get_node_attributes(ref_graph, "resname")
