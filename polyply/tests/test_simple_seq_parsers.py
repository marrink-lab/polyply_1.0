# Copyright 2021 University of Groningen
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
"""
Test that sequence files are properly read.
"""
from pathlib import Path
import pytest
import networkx as nx
from polyply import TEST_DATA
from polyply.src.meta_molecule import MetaMolecule
from .example_fixtures import example_meta_molecule
from polyply.src.simple_seq_parsers import (_identify_nucleotypes,
                                            _monomers_to_linear_nx_graph,
                                            _parse_plain,
                                            FileFormatError)

@pytest.mark.parametrize('comments, DNA, RNA', (
    # single DNA comment
      (["DNA lorem ipsum"],
       True,
       False
      ),
    # single RNA comment
      (["RNA lorem ipsum"],
       False,
       True
      ),
    # single DNA comment multiple lines
      (["lorem ipsum", "random line DNA", "DNA another line"],
       True,
       False
      ),
    # single RNA comment multiple lines
      (["lorem ipsum", "random line RNA", "RNA another line"],
       False,
       True
      ),
     ))
def test_identify_nucleotypes(comments, DNA, RNA):
    out_DNA, out_RNA = _identify_nucleotypes(comments)
    assert out_DNA == DNA
    assert out_RNA == RNA

@pytest.mark.parametrize('comments', (
    # both DNA and RNA are defined
      ["DNA RNA ipsum", "another line"],
    # neither DNA and RNA are defined
      ["lorem ipsum", "one more line"]
     ))
def test_identify_nucleotypes_fail(comments):
    with pytest.raises(FileFormatError):
        _identify_nucleotypes(comments)

def _node_match(nodeA, nodeB):
    resname = nodeA["resname"] == nodeB["resname"]
    resid = nodeA["resid"] == nodeB["resid"]
    return resname & resid

def test_monomers_to_linear_nx_graph(example_meta_molecule):
    monomers = ["A", "B", "A"]
    seq_graph = _monomers_to_linear_nx_graph(monomers)
    assert nx.is_isomorphic(seq_graph, example_meta_molecule, node_match=_node_match)

@pytest.mark.parametrize('extension, ', (
      "txt",
      "ig",
      "fasta"
     ))
def test_sequence_parses(extension):
    filepath = Path(TEST_DATA + "/simple_seq_files/test."+ extension)
    seq_graph = MetaMolecule.parsers[extension](filepath)
    monomers = ["DA5", "DT", "DC", "DG", "DT", "DA", "DC", "DA", "DT3"]
    ref_graph = _monomers_to_linear_nx_graph(monomers)
    assert nx.is_isomorphic(seq_graph, ref_graph, node_match=_node_match)

def test_ig_cirle():
    filepath = Path(TEST_DATA + "/simple_seq_files/test_circle.ig")
    seq_graph = MetaMolecule.parsers["ig"](filepath)
    monomers = ["DA", "DT", "DC", "DG", "DT", "DA", "DC", "DA", "DT"]
    ref_graph = _monomers_to_linear_nx_graph(monomers)
    ref_graph.add_edge(0, 8)
    assert seq_graph.edges[(0, 8)]["linktype"] == "circle"
    assert nx.is_isomorphic(seq_graph,
                            ref_graph,
                            node_match=_node_match)

def test_ig_termination_fail():
    filepath = Path(TEST_DATA + "/simple_seq_files/test_fail.ig")
    with pytest.raises(FileFormatError):
        seq_graph = MetaMolecule.parsers["ig"](filepath)

@pytest.mark.parametrize('extension, ', (
      "ig",
      "fasta"
     ))
def test_sequence_parses_RNA(extension):
    filepath = Path(TEST_DATA + "/simple_seq_files/test_RNA."+ extension)
    seq_graph = MetaMolecule.parsers[extension](filepath)
    monomers = ["A5", "U", "C", "G", "U", "A", "C", "A", "U3"]
    ref_graph = _monomers_to_linear_nx_graph(monomers)
    assert nx.is_isomorphic(seq_graph, ref_graph, node_match=_node_match)

def test_unkown_nucleotype_error():
    with pytest.raises(IOError):
        lines = ["AABBBCCTG"]
        _parse_plain(lines, DNA=True, RNA=False)
