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

from pathlib import Path
import textwrap
import argparse
import pytest
import networkx as nx
import vermouth.forcefield
import vermouth.molecule
import vermouth.gmx.itp_read
from polyply import gen_params, TEST_DATA, MetaMolecule
from polyply.src.graph_utils import find_missing_edges

class TestGenParams():
    @staticmethod
    @pytest.mark.parametrize('args_in, ref_file', (
        (["-f", TEST_DATA + "/gen_params/input/PEO.martini.3.itp"
         ,"-seq", "PEO:10", "-name", "PEO", "-o",
         TEST_DATA + "/gen_params/output/PEO_out.itp"],
         TEST_DATA + "/gen_params/ref/PEO_10.itp"),
        (["-f", TEST_DATA + "/gen_params/input/PS.martini.2.itp",
         "-seqf", TEST_DATA + "/gen_params/input/PS.json",
         "-name", "PS",
         "-o", TEST_DATA + "/gen_params/output/PS_out.itp"]
         ,
         TEST_DATA + "/gen_params/ref/PS_10.itp"),
        (["-f", TEST_DATA + "/gen_params/input/P3HT.martini.2.itp",
         "-seq", "P3HT:10",
         "-name", "P3HT",
         "-o", TEST_DATA + "/gen_params/output/P3HT_out.itp"],
         TEST_DATA + "/gen_params/ref/P3HT_10.itp"),
        (["-f", TEST_DATA + "/gen_params/input/PPI.ff",
         "-seqf", TEST_DATA + "/gen_params/input/PPI.json",
         "-name", "PPI",
         "-o", TEST_DATA + "/gen_params/output/PPI_out.itp"],
         TEST_DATA + "/gen_params/ref/G3.itp"),
        (["-f", TEST_DATA + "/gen_params/input/test.ff",
         "-seq", "N1:1", "N2:1", "N1:1", "N2:1", "N3:1",
         "-name", "test",
         "-o", TEST_DATA + "/gen_params/output/test_out.itp"],
         TEST_DATA + "/gen_params/ref/test_rev.itp"),
        # check if edge attributes are parsed and properly applied
        (["-f", TEST_DATA+"/gen_params/input/test_edge_attr.ff",
         "-seqf", TEST_DATA + "/gen_params/input/test_edge_attr.json",
         "-name", "test",
         "-o", TEST_DATA + "/gen_params/output/test_edge_attr_out.itp"],
         TEST_DATA + "/gen_params/ref/test_edge_attr_ref.itp"),
        # check if nodes can be removed
        (["-f", TEST_DATA+"/gen_params/input/removal.ff",
         "-seq", "PEO:3",
         "-name", "test",
         "-o", TEST_DATA + "/gen_params/output/removal.itp"],
         TEST_DATA + "/gen_params/ref/removal.itp")
        ))
    def test_gen_params(args_in, ref_file):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument('-name', required=True, type=str, dest="name",
                            help="name of the final molecule")
        file_group = parser.add_argument_group('Input and output files')
        file_group.add_argument('-f', dest='inpath', required=False, type=Path,
                                help='Input file (ITP|FF)', nargs="*")
        file_group.add_argument('-o', dest='outpath', type=Path,
                                help='Output ITP (ITP)')
        file_group.add_argument('-seq', dest='seq', required=False, nargs='+',
                                type=str, help='linear sequence')
        file_group.add_argument('-seqf', dest='seq_file', required=False, type=Path,
                                help='linear sequence')
        ff_group = parser.add_argument_group('Force field selection')
        ff_group.add_argument('-lib', dest='lib', required=False, type=str,
                              help='force-fields to include from library', nargs='*')

        args = parser.parse_args(args_in)
        gen_params(**vars(args))

        force_field = vermouth.forcefield.ForceField(name='test_ff')

        for path_name in [args.outpath, ref_file]:
            with open(path_name, 'r') as _file:
                lines = _file.readlines()
            vermouth.gmx.itp_read.read_itp(lines, force_field)

        ref_name = args.name + "ref"

        #1. Check that all nodes and attributes are the same
        assert set(force_field.blocks[ref_name].nodes) == set(force_field.blocks[args.name].nodes)
        for node in force_field.blocks[ref_name].nodes:
            ref_attrs = nx.get_node_attributes(force_field.blocks[ref_name], node)
            new_attrs = nx.get_node_attributes(force_field.blocks[args.name], node)
            assert new_attrs == ref_attrs

        #2. Check that all interactions are the same
        int_types_ref = force_field.blocks[ref_name].interactions.keys()
        int_types_new = force_field.blocks[args.name].interactions.keys()
        assert int_types_ref == int_types_new

        for key in force_field.blocks[ref_name].interactions:
            for term in force_field.blocks[ref_name].interactions[key]:
                assert term in force_field.blocks[args.name].interactions[key]

def test_find_missing_links():
    fname = TEST_DATA + "/gen_params/ref/P3HT_10.itp"
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
