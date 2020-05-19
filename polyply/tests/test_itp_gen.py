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
from polyply import gen_itp

class TestGenItp():
    @staticmethod
    @pytest.mark.parametrize('arg_string, ref_file', (
        ("""-f test_data/gen_itp/input/PEO.martini.3.itp
         -seq PEO:10
         -name PEO
         -o test_data/gen_itp/output/PEO_out.itp
         """,
         "test_data/gen_itp/ref/PEO_10.itp"),
        ("""-f test_data/gen_itp/input/PS.martini.2.itp
         -seqf test_data/gen_itp/input/PS.json
         -name PS
         -o test_data/gen_itp/output/PS_out.itp
         """,
         "test_data/gen_itp/ref/PS_10.itp"),
        ("""-f test_data/gen_itp/input/P3HT.martini.2.itp
         -seq P3HT:10
         -name P3HT
         -o test_data/gen_itp/output/P3HT_out.itp
         """,
         "test_data/gen_itp/ref/P3HT_10.itp"),
        ("""-f test_data/gen_itp/input/PPI.ff
         -seqf test_data/gen_itp/input/PPI.json
         -name PPI
         -o test_data/gen_itp/output/PPI_out.itp
         """,
         "test_data/gen_itp/ref/G3.itp")
        ))
    def test_gen_itp(arg_string, ref_file):
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
        file_group.add_argument('-seq', dest='seq', required=False, type=str,
                                help='linear sequence')
        file_group.add_argument('-seqf', dest='seq_file', required=False, type=Path,
                                help='linear sequence')
        ff_group = parser.add_argument_group('Force field selection')
        ff_group.add_argument('-lib', dest='lib', required=False, type=str,
                              help='force-fields to include from library')
        ff_group.add_argument('-ff-dir', dest='extra_ff_dir', action='append',
                              type=Path, default=[],
                              help='Additional repository for custom force fields.')
        ff_group.add_argument('-list-lib', action='store_true', dest='list_ff',
                              help='List all known force fields, and exit.')
        ff_group.add_argument('-list-blocks', action='store_true', dest='list_blocks',
                              help='List all Blocks known to the'
                              ' force field, and exit.')
        ff_group.add_argument('-list-links', action='store_true', dest='list_links',
                              help='List all Links known to the'
                              ' force field, and exit.')

        arg_in = [arg.lstrip() for arg in textwrap.dedent(arg_string).split()]
        args = parser.parse_args(arg_in)
        gen_itp(args)

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
            print(key)
            for term in force_field.blocks[ref_name].interactions[key]:
                print(term)
                assert term in force_field.blocks[args.name].interactions[key]
