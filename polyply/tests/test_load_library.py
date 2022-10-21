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
Test linear algebra aux functions.
"""

import textwrap
import pytest
from pathlib import Path
import numpy as np
import os
import math
from contextlib import contextmanager
import networkx as nx
import vermouth
import polyply
from polyply import TEST_DATA
from polyply.src.topology import Topology
from polyply.src.load_library import FORCE_FIELD_PARSERS
from polyply.src.load_library import _resolve_lib_paths
from polyply.src.load_library import read_ff_from_files
from polyply.src.load_library import load_build_options
from polyply.src.load_library import check_extensions_ff
from polyply.src.load_library import check_extensions_bld


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

@pytest.mark.parametrize("ff_file, expectation", [
    [[Path('forcefield.ff')], nullcontext()],
    [[Path('forcefield.bld')], pytest.raises(IOError)],
])
def test_check_extensions_ff(ff_file, expectation):
    with expectation as e:
        check_extensions_ff(ff_file)

@pytest.mark.parametrize("bld_file, expectation", [
    [[Path('forcefield.ff')], pytest.raises(IOError)],
    [[Path('forcefield.bld')], nullcontext()],
])
def test_check_extensions_bld(bld_file, expectation):
    with expectation as e:
        check_extensions_bld(bld_file)

def test_read_ff_from_files():
    name = "ff"
    force_field = vermouth.forcefield.ForceField(name)
    lib_files = _resolve_lib_paths([name], TEST_DATA, FORCE_FIELD_PARSERS.keys())
    read_ff_from_files(lib_files, force_field)

    # Check if .ff files were parsed
    assert force_field.blocks
    assert force_field.links

def test_read_build_options_from_files():

    topfile = 'topology_test/system.top'
    bldfile = 'topology_test/test.bld'
    lib_names = ['2016H66']
    toppath = os.path.join(TEST_DATA, topfile)
    topology = Topology.from_gmx_topfile(name='test', path=toppath)
    topology.preprocess()
    bldpath = os.path.join(TEST_DATA, bldfile)
    load_build_options(topology, lib_names, bldpath)

    # check if build files are parsed
    assert topology.volumes == {'PMMA': 1.0}
    molecule = topology.molecules[0]
    assert molecule.templates
