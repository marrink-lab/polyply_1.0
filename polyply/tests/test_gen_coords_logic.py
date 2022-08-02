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
"""
This collection of tests checks some basic I/O workflows of the gen_coords
function. For example, that it runs through when no coordinates are requested
to be generated.
"""
import logging
import pathlib
import pytest
import numpy as np
import networkx as nx
from vermouth.gmx.gro import read_gro
from vermouth.graph_utils import make_residue_graph
import polyply
from polyply import TEST_DATA, gen_coords

def test_no_positions_generated(tmp_path, monkeypatch):
    """
    All positions are defined none have to be created. Should run through without
    errors and preserve all positions given in the input to rounding accuracy.
    """
    monkeypatch.chdir(tmp_path)
    top_file = TEST_DATA + "/topology_test/system.top"
    pos_file = TEST_DATA + "/topology_test/complete.gro"
    out_file = tmp_path / "out.gro"
    gen_coords(toppath=top_file,
               coordpath=pos_file,
               outpath=out_file,
               name="test",
               box=np.array([10, 10, 10])
               )
    molecule_out = read_gro(out_file, exclude=())
    molecule_in = read_gro(pos_file, exclude=())
    for node in molecule_out.nodes:
        assert np.all(molecule_out.nodes[node]['position'] ==
                      molecule_in.nodes[node]['position'])

def test_backmap_only(tmp_path, monkeypatch):
    """
    Only meta_mol positions are defined so others have to be backmapped.
    Should run through without errors and COG of residues should be the
    same as they have been put in.
    """
    monkeypatch.chdir(tmp_path)
    top_file = TEST_DATA + "/topology_test/system.top"
    pos_file = TEST_DATA + "/topology_test/cog.gro"
    out_file = tmp_path / "out.gro"
    gen_coords(toppath=top_file,
               coordpath_meta=pos_file,
               outpath=out_file,
               name="test",
               box=np.array([11, 11, 11])
               )
    molecule_out = read_gro(out_file, exclude=())
    for node in molecule_out.nodes:
        assert "position" in molecule_out.nodes[node]

    meta_in = read_gro(pos_file, exclude=())
    meta_out = make_residue_graph(molecule_out)
    for node in meta_out.nodes:
        nodes_pos = nx.get_node_attributes(meta_out.nodes[node]['graph'], 'position')
        nodes_pos = np.array(list(nodes_pos.values()))
        res_pos = np.average(nodes_pos, axis=0)
        ref_pos = meta_in.nodes[node]['position']
        # tolerance comes from finite tolerance in gro file
        assert np.allclose(res_pos, ref_pos, atol=0.0009)

def test_warning_partial_metamol_coords(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    top_file = TEST_DATA + "/topology_test/system.top"
    pos_file = TEST_DATA + "/topology_test/cog_missing.gro"
    out_file = tmp_path / "out.gro"

    with caplog.at_level(logging.WARNING):
        gen_coords(toppath=top_file,
                   coordpath_meta=pos_file,
                   outpath=out_file,
                   name="test",
                   box=np.array([11, 11, 11])
                   )
        print(caplog.records)
        for record in caplog.records:
            assert record.levelname == "WARNING"
            break
        else:
            assert False
