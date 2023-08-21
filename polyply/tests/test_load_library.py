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
Test library loading behaviour.
"""

import os
import logging
import pytest
from pathlib import Path
from contextlib import contextmanager
import vermouth
from polyply import TEST_DATA
from polyply.src.logging import LOGGER
from polyply.src.topology import Topology
from polyply.src.load_library import FORCE_FIELD_PARSERS, BUILD_FILE_PARSERS
from polyply.src.load_library import _resolve_lib_files, get_parser
from polyply.src.load_library import load_build_files, read_options_from_files


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

@pytest.mark.parametrize("file_, file_parser, ignore_bld_files, expectation", [
    [Path('forcefield.ff'), FORCE_FIELD_PARSERS, True, nullcontext()],
    [Path('forcefield.bld'), FORCE_FIELD_PARSERS, True, nullcontext()],
    [Path('forcefield.ff'), BUILD_FILE_PARSERS, False, pytest.raises(IOError)],
    [Path('forcefield.bld'), BUILD_FILE_PARSERS, False, nullcontext()],
])
def test_get_parser(file_, file_parser, ignore_bld_files, expectation):
    with expectation as e:
        parser = get_parser(file_, file_parser, ignore_bld_files)


def test_read_ff_from_files(caplog):

    name = "ff"
    force_field = vermouth.forcefield.ForceField(name)
    lib_files = _resolve_lib_files([name], TEST_DATA)
    user_files = []
    all_files = [lib_files, user_files]

    # Check if warning is thrown for unknown file
    loglevel = getattr(logging, 'WARNING')
    LOGGER.setLevel(loglevel)

    msg = "File with unknown extension txt found in force field library."
    with caplog.at_level(loglevel):
        read_options_from_files(all_files, force_field, FORCE_FIELD_PARSERS)
        for record in caplog.records:
            if record.message == msg:
                break
        else:
            assert False

    # Check if .ff files were parsed
    assert force_field.blocks
    assert force_field.links


def test_read_build_options_from_files():

    topfile = Path('topology_test/system.top')
    bldfile = Path('topology_test/test.bld')
    lib_name = '2016H66'
    toppath = Path(TEST_DATA).joinpath(topfile)
    topology = Topology.from_gmx_topfile(name='test', path=toppath)
    topology.preprocess()

    user_files = [Path(TEST_DATA).joinpath(bldfile)]
    load_build_files(topology, lib_name, user_files)

    # check if build files are parsed
    assert topology.volumes == {'PMMA': 1.0}
    molecule = topology.molecules[0]
    assert molecule.templates
