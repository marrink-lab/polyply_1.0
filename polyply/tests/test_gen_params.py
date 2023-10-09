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
import os
from pathlib import Path
import textwrap
import argparse
import logging
import pytest
import networkx as nx
import vermouth.forcefield
import vermouth.molecule
import vermouth.gmx.itp_read
from polyply import gen_params, TEST_DATA, MetaMolecule
from polyply.src.graph_utils import find_missing_edges
from polyply.src.logging import LOGGER

@pytest.mark.parametrize('inpath, seq, seqf, name, ref_file', (
    ([TEST_DATA / "gen_params" / "input" / "PEO.martini.3.itp"],
     ["PEO:10"],
     None,
     "PEO",
     TEST_DATA / "gen_params" / "ref" / "PEO_10.itp"),
    ([TEST_DATA / "gen_params"/ "input"/"PS.martini.2.itp"],
     None,
     TEST_DATA / "gen_params" / "input" / "PS.json",
     "PS",
     TEST_DATA / "gen_params" / "ref" / "PS_10.itp"),
    ([TEST_DATA / "gen_params" / "input" / "P3HT.martini.2.itp"],
     ["P3HT:10"],
     None,
     "P3HT",
     TEST_DATA / "gen_params" / "ref" / "P3HT_10.itp"),
    ([TEST_DATA / "gen_params" / "input" / "PPI.ff"],
     None,
     TEST_DATA / "gen_params" / "input" / "PPI.json",
     "PPI",
     TEST_DATA / "gen_params" / "ref" / "G3.itp"),
    ([TEST_DATA / "gen_params" / "input" / "test.ff"],
     ["N1:1", "N2:1", "N1:1", "N2:1", "N3:1"],
     None,
     "test",
     TEST_DATA / "gen_params" / "ref" / "test_rev.itp"),
    # check if edge attributes are parsed and properly applied
    ([TEST_DATA / "gen_params" / "input" / "test_edge_attr.ff"],
     None,
     TEST_DATA / "gen_params" / "input" / "test_edge_attr.json",
     "test",
     TEST_DATA / "gen_params" / "ref" / "test_edge_attr_ref.itp"),
    # check if nodes can be removed
    ([TEST_DATA / "gen_params" / "input" / "removal.ff"],
     ["PEO:3"],
     None,
     "test",
     TEST_DATA / "gen_params" / "ref" / "removal.itp")
    ))
def test_gen_params(tmp_path, inpath, seq, seqf, name, ref_file):
    os.chdir(tmp_path)
    gen_params(inpath=inpath, seq=seq, seq_file=seqf, name=name)

    force_field = vermouth.forcefield.ForceField(name='test_ff')

    for path_name in [tmp_path / "polymer.itp", ref_file]:
        with open(path_name, 'r') as _file:
            lines = _file.readlines()
        vermouth.gmx.itp_read.read_itp(lines, force_field)

    ref_name = name + "ref"

    #1. Check that all nodes and attributes are the same
    assert set(force_field.blocks[ref_name].nodes) == set(force_field.blocks[name].nodes)
    for node in force_field.blocks[ref_name].nodes:
        ref_attrs = nx.get_node_attributes(force_field.blocks[ref_name], node)
        new_attrs = nx.get_node_attributes(force_field.blocks[name], node)
        assert new_attrs == ref_attrs

    #2. Check that all interactions are the same
    int_types_ref = force_field.blocks[ref_name].interactions.keys()
    int_types_new = force_field.blocks[name].interactions.keys()
    assert int_types_ref == int_types_new

    for key in force_field.blocks[ref_name].interactions:
        for term in force_field.blocks[ref_name].interactions[key]:
            assert term in force_field.blocks[name].interactions[key]

def test_find_missing_links():
    fname = TEST_DATA / "gen_params" / "ref" / "P3HT_10.itp"
    ff = vermouth.forcefield.ForceField("test")
    meta_mol = MetaMolecule.from_itp(ff, fname, "P3HTref")
    meta_mol.molecule.remove_edge(39, 45) # resid 7,8
    meta_mol.molecule.remove_edge(15, 21) # resid 3,4
    missing = list(find_missing_edges(meta_mol, meta_mol.molecule))
    assert len(missing) == 2
    for edge, ref in zip(missing, [(3, 4), (7, 8)]):
        assert edge["resA"] == "P3HTref"
        assert edge["resB"] == "P3HTref"
        assert edge["idxA"] == ref[0]
        assert edge["idxB"] == ref[1]

@pytest.mark.parametrize('warn_type, ffobject',
                         (('INFO', 'link'),
                          ('INFO', 'block'),
                          ('WARNING', 'link'),
                          ('WARNING', 'block'),
                          ('ERROR', 'link'),
                          ('ERROR', 'block')))
def test_print_log_warnings(tmp_path, monkeypatch, caplog, warn_type, ffobject):
    """
    Quick test to make sure that the logging warnings propagate to the
    gen_params output.
    """
    # change to temporary direcotry
    monkeypatch.chdir(tmp_path)

    # get input file from test data
    infile = TEST_DATA / Path(f"gen_params/logging/{warn_type}_{ffobject}.ff")

    # set loglevel
    loglevel = getattr(logging, warn_type)
    LOGGER.setLevel(loglevel)

    # capture logging messages
    with caplog.at_level(loglevel):
        gen_params(name="polymer",
                   outpath=Path("polymer.itp"),
                   inpath=[infile],
                   lib=None,
                   seq=["test:5"],
                   seq_file=None)

        assert f"This is a {warn_type}." in [record.getMessage() for record in caplog.records if record.levelname == warn_type]
