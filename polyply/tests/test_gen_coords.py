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
import pytest
import textwrap
from vermouth.forcefield import ForceField
import polyply
from polyply.src.topology import Topology
from polyply.src.top_parser import read_topology
from .example_fixtures import example_meta_molecule

@pytest.fixture
def example_topology():
    top_lines ="""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 45.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0  1   GLY    BB   1 0.00     45
    2    N0  1   GLY    SC1  1 0.00     45
    3    N0  1   GLY    SC2  1 0.00     45
    4    N0  2   GLU    BB   2 0.00     45
    5    N0  2   GLU    SC1  2 0.00     45
    6    N0  2   GLU    SC2  2 0.00     45
    [ bonds ]
    1    2    1  0.47 2000
    2    3    1  0.47 2000
    1    4    1  0.47 2000
    4    5    1  0.47 2000
    4    6    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0  1   ASP    BB   1 0.00
    2    N0  1   ASP    SC1  1 0.00
    3    N0  1   ASP    SC2  1 0.00
    [ bonds ]
    1    2    1  0.47 2000
    2    3    1  0.47 2000
    [ system ]
    test system
    [ molecules ]
    testA 2
    testB 1
    """
    lines = textwrap.dedent(top_lines)
    lines = lines.splitlines()
    force_field = ForceField("test")
    topology = Topology(force_field)
    read_topology(lines=lines, topology=topology, cwdir="./")
    topology.preprocess()
    topology.volumes = {"GLY": 0.53, "GLU": 0.67, "ASP": 0.43}
    return topology

@pytest.mark.parametrize('start_nodes, expected', (
   # no starting nodes defined
   ([],
   {0:None, 1:None, 2:None}
   ),
   # starting node for each molecule
   (
   ["testA-GLU", "testB-ASP"],
   {0:1, 1:1, 2:0}
   ),
   # starting node different for first two molecules
   (
   ["testA#0-GLU", "testB#1-GLY" ,"testB-ASP"],
   {0:1, 1:0, 2:0}
   )))
def test_find_starting_node_from_spec(example_topology, start_nodes, expected):
   """<mol_name>#<mol_idx>-<resname>#<resid>"""
   result = polyply.src.gen_coords.find_starting_node_from_spec(example_topology,
                                                                start_nodes)
   assert result == expected

def test_check_molecule(example_meta_molecule):
    # test if check passes for connected molecule
    polyply.src.gen_coords._check_molecules([example_meta_molecule])
    # check that if we delte an edge an error is raised
    example_meta_molecule.remove_edge(0, 1)
    with pytest.raises(IOError):
        polyply.src.gen_coords._check_molecules([example_meta_molecule])
