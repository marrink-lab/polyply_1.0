# Copyright 2022 University of Groningen
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
Test the charge modification functions used in itp_to_ff.
"""
import textwrap
import pytest
from pathlib import Path
import networkx as nx
import vermouth
import polyply
from polyply.src.charges import balance_charges
@pytest.mark.parametrize('charges, target',(
    ({0: 0.2, 1: -0.4, 2: 0.23, 3: 0.001},
     0.0,),
    ({0: 0.6, 1: -0.2, 2: 0.5, 3: 0.43},
     0.5,),
    ({0: -0.633, 1: -0.532, 2: 0.512, 3: 0.0},
     -0.6,),
))
def test_balance_charges(charges, target):
    lines = """
    [ moleculetype ]
    test 1
    [ atoms ]
    1 P4 1 GLY BB  1
    2 P3 1 GLY SC1 2
    3 P2 1 ALA SC2 3
    4 P2 1 ALA SC3 3
    [ bonds ]
    1 2 1 0.2 100
    2 3 1 0.6 700
    3 4 1 0.2 700
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    block = ff.blocks['test']
    nx.set_node_attributes(block, charges, 'charge')
    balance_charges(block, topology=None, charge=target, tol=10**-5, decimals=5)
    new_charges = nx.get_node_attributes(block, 'charge')
    assert pytest.approx(sum(new_charges.values()),abs=0.0001) == target
