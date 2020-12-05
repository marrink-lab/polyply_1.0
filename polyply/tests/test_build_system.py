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
Test that force field files are properly read.
"""
import pytest
import textwrap
import numpy as np
from vermouth.forcefield import ForceField
import polyply
from polyply.src.topology import Topology
from polyply.src.top_parser import read_topology
from polyply.src.build_system import (_compute_box_size,
                                      _filter_by_molname,
                                      BuildSystem
                                     )

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

@pytest.mark.parametrize('density, result', (
    (1000.0,
     1.0387661572640916
    ),
    (600.0,
     1.2315934632345062
     )))
def test_compute_box_size(example_topology, density, result):
    top = example_topology
    assert np.isclose(_compute_box_size(top, density), result)


def test_compute_box_size_error(example_topology):
    top = example_topology
    # clear the atomtypes to see if error is raised
    # when no mass can be found
    top.atom_types = {}
    with pytest.raises(KeyError):
        _compute_box_size(top, 1000)

@pytest.mark.parametrize('ignore', (
    ["test_A"],
    ["test_A", "test_B"],
     ))
def test_filer_by_molname(example_topology, ignore):
    molecules = _filter_by_molname(example_topology.molecules, ignore)
    for molecule in molecules:
        assert molecule.mol_name not in ignore


@pytest.mark.parametrize('box, density, grid_spacing, grid, expected_grid_shape, expected_box', (
   # grid and box are generated automatically
   (None,
    1000,
    0.2,
    None,
    (216, 3),
    np.array([1.03877, 1.03877, 1.03877])
   ),
   # box is provided and grid is generated automatically
   (np.array([5.0, 5.0, 5.0]),
    1000,
    0.2,
    None,
    (15625, 3),
    np.array([5.0, 5.0, 5.0]),
   ),
   # box is provided and grid also nothing is generated
   (np.array([5.0, 5.0, 5.0]),
    1000,
    0.2,
    np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    (2, 3),
    np.array([5.0, 5.0, 5.0]),
   ),
   ))
def test_init(example_topology,
             box, density,
             grid_spacing, grid,
             expected_grid_shape,
             expected_box):

   processor = BuildSystem(example_topology,
                           start_dict={},
                           box=box,
                           density=density,
                           grid=grid,
                           grid_spacing=grid_spacing,
                           )
   assert processor.box_grid.shape == expected_grid_shape
   assert all(processor.box == expected_box)

def test_rw_arg_failure():
    with pytest.raises(TypeError):
         BuildSystem(example_topology,
                     density=1000,
                     start_dict={},
                     random_argument=10)

@pytest.mark.parametrize("""positions, box,
                          starting_grid,
                          starting_nodes,
                          """, (
   # no inital starting positoins
   ([],
    np.array([5., 5., 5.]),
    np.array([[2.0, 2.0, 2.0],
              [3.0, 3.0, 3.0],
              [0.5, 0.5, 0.5],
              [4.0, 4.0, 4.0]]),
    {0:0, 1:0, 2:0}
   ),
   # skip first molecule
   (np.array([[1.25, 1.00, 1.00],
              [1.75, 1.00, 1.00],
             ]),
    np.array([5., 5., 5.]),
    np.array([[2.0, 2.0, 2.0],
              [3.0, 3.0, 3.0],
              [0.5, 0.5, 0.5],
              [4.0, 4.0, 4.0]]),
   {0:0, 1:0, 2:0}
   ),
   # partial starting positions
   (np.array([[1.25, 1.00, 1.00]]),
    np.array([5., 5., 5.]),
    np.array([[2.0, 2.0, 2.0],
              [3.0, 3.0, 3.0],
              [0.5, 0.5, 0.5],
              [4.0, 4.0, 4.0]]),
    {0:0, 1:0, 2:0}
   ),
   # scramble starting nodes
   ([],
    np.array([5., 5., 5.]),
    np.array([[2.0, 2.0, 2.0],
              [3.0, 3.0, 3.0],
              [0.5, 0.5, 0.5],
              [4.0, 4.0, 4.0]]),
    {0:1, 1:0, 2:0}
   ),
   ))
def test_build_system(positions,
                      example_topology,
                      box,
                      starting_grid,
                      starting_nodes):

   total = 0
   for mol in example_topology.molecules:
       for node in mol.nodes:
           try:
               mol.nodes[node]["position"] = positions[total]
               mol.nodes[node]["build"] = False
           except IndexError:
               mol.nodes[node]["build"] = True
           total += 1


   processor = BuildSystem(example_topology,
                           density=1000,
                           start_dict=starting_nodes,
                           box=box,
                           grid=starting_grid,
                           )
   processor.run_system(example_topology)
   total = 0
   for mol_idx, mol in enumerate(example_topology.molecules):
       for node in mol.nodes:
           if total < len(positions):
              assert all(mol.nodes[node]["position"] == positions[total])
           elif node == starting_nodes[mol_idx]:
              assert mol.nodes[node]["position"] in starting_grid
           else:
              assert "position" in mol.nodes[node]
           total += 1
