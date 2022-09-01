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
import numpy as np
import math
import networkx as nx
import vermouth
import polyply
from polyply import TEST_DATA
from polyply.src.load_library import _resolve_lib_paths
from polyply.src.load_library import read_ff_from_files
from polyply.src.load_library import FORCE_FIELD_PARSERS

def test_read_ff_from_files():
    name = "ff"
    force_field = vermouth.forcefield.ForceField(name)
    lib_files = _resolve_lib_paths([name], TEST_DATA, FORCE_FIELD_PARSERS.keys())
    read_ff_from_files(lib_files, force_field)

    # Check if .ff files were parsed
    assert force_field.blocks
    assert force_field.links

