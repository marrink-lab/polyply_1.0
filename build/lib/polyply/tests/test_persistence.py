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
Test that build files are properly read.
"""

import logging
import pytest
import textwrap
import math
import numpy as np
import networkx as nx
from vermouth.forcefield import ForceField
import polyply
from polyply.src.topology import Topology
from polyply.src.top_parser import read_topology
from polyply.src.nonbond_engine import NonBondEngine
from polyply.src.build_file_parser import PersistenceSpecs
from polyply.src.persistence import sample_end_to_end_distances


@pytest.fixture
def topology():
    top_lines ="""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 72.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0   1   GLY    BB   1 0.00     45
    2    N0   2   GLY    SC1  1 0.00     45
    3    N0   3   GLY    SC2  1 0.00     45
    4    N0   4   GLU    BB   2 0.00     45
    5    N0   5   GLU    SC1  2 0.00     45
    6    N0   6   GLU    SC2  2 0.00     45
    7    N0   7   GLU    SC2  2 0.00     45
    8    N0   8   GLU    SC2  2 0.00     45
    9    N0   9   GLU    SC2  2 0.00     45
   10    N0  10   GLU    SC2  2 0.00     45
   11    N0  11   GLU    SC2  2 0.00     45
   12    N0  12   GLU    SC2  2 0.00     45
   13    N0  13   GLU    SC2  2 0.00     45
   14    N0  14   GLU    SC2  2 0.00     45
   15    N0  15   GLU    SC2  2 0.00     45
   16    N0  16   GLU    SC2  2 0.00     45
   17    N0  17   GLU    SC2  2 0.00     45
   18    N0  18   GLU    SC2  2 0.00     45
   19    N0  19   GLU    SC2  2 0.00     45
   20    N0  20   GLU    SC2  2 0.00     45
   21    N0  21   GLU    SC2  2 0.00     45
   22    N0  22   GLU    SC2  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    6     7    1  0.47 2000
    7     8    1  0.47 2000
    8     9    1  0.47 2000
    9    10    1  0.47 2000
   10    11    1  0.47 2000
   11    12    1  0.47 2000
   12    13    1  0.47 2000
   13    14    1  0.47 2000
   14    15    1  0.47 2000
   15    16    1  0.47 2000
   16    17    1  0.47 2000
   17    18    1  0.47 2000
   18    19    1  0.47 2000
   19    20    1  0.47 2000
   20    21    1  0.47 2000
   21    22    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0   1   ASP    BB   1 0.00     45
    2    N0   2   GLY    SC1  1 0.00     45
    3    N0   3   ASP    SC2  1 0.00     45
    4    N0   4   ASP    BB   2 0.00     45
    5    N0   5   GLY    SC1  2 0.00     45
    6    N0   6   ASP    SC2  2 0.00     45
    7    N0   7   ASP    SC2  2 0.00     45
    8    N0   8   GLU    SC2  2 0.00     45
    9    N0   9   GLU    SC2  2 0.00     45
   10    N0  10   GLY    SC2  2 0.00     45
   11    N0  11   ASP    SC2  2 0.00     45
   12    N0  12   GLU    SC2  2 0.00     45
   13    N0  13   ASP    SC2  2 0.00     45
   14    N0  14   ASP    SC2  2 0.00     45
   15    N0  15   GLU    SC2  2 0.00     45
   16    N0  16   ASP    SC2  2 0.00     45
   17    N0  17   ASP    SC2  2 0.00     45
   18    N0  18   GLY    SC2  2 0.00     45
   19    N0  19   ASP    SC2  2 0.00     45
   20    N0  20   ASP    SC2  2 0.00     45
   21    N0  21   GLY    SC2  2 0.00     45
   22    N0  22   ASP    SC2  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    6     7    1  0.47 2000
    7     8    1  0.47 2000
    8     9    1  0.47 2000
    9    10    1  0.47 2000
   10    11    1  0.47 2000
   11    12    1  0.47 2000
   12    13    1  0.47 2000
   13    14    1  0.47 2000
   14    15    1  0.47 2000
   15    16    1  0.47 2000
   16    17    1  0.47 2000
   17    18    1  0.47 2000
   18    19    1  0.47 2000
   19    20    1  0.47 2000
   20    21    1  0.47 2000
   21    22    1  0.47 2000
    [ moleculetype ]
    testC 1
    [ atoms ]
    1    N0   1   GLY    BB   1 0.00     45
    2    N0   2   GLY    SC1  1 0.00     45
    3    N0   3   GLY    SC2  1 0.00     45
    4    N0   4   GLU    BB   2 0.00     45
    5    N0   5   GLU    SC1  2 0.00     45
    6    N0   6   GLU    SC2  2 0.00     45
    7    N0   7   GLU    SC2  2 0.00     45
    8    N0   8   GLU    SC2  2 0.00     45
    9    N0   9   GLU    SC2  2 0.00     45
   10    N0  10   GLU    SC2  2 0.00     45
   11    N0  11   GLU    SC2  2 0.00     45
   12    N0  12   GLU    SC2  2 0.00     45
   13    N0  13   GLU    SC2  2 0.00     45
   14    N0  14   GLU    SC2  2 0.00     45
   15    N0  15   GLU    SC2  2 0.00     45
   16    N0  16   GLU    SC2  2 0.00     45
   17    N0  17   GLU    SC2  2 0.00     45
   18    N0  18   GLU    SC2  2 0.00     45
   19    N0  19   GLU    SC2  2 0.00     45
   20    N0  20   GLU    SC2  2 0.00     45
   21    N0  21   GLU    SC2  2 0.00     45
   22    N0  22   GLU    SC2  2 0.00     45
   23    N0  23   GLU    SC2  2 0.00     45
   24    N0  24   GLU    SC2  2 0.00     45
   25    N0  25   GLU    SC2  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    6     7    1  0.47 2000
    7     8    1  0.47 2000
    8     9    1  0.47 2000
    9    10    1  0.47 2000
   10    11    1  0.47 2000
   11    12    1  0.47 2000
   12    13    1  0.47 2000
   13    14    1  0.47 2000
   14    15    1  0.47 2000
   15    16    1  0.47 2000
   16    17    1  0.47 2000
   17    18    1  0.47 2000
   18    19    1  0.47 2000
   19    20    1  0.47 2000
   20    21    1  0.47 2000
   21    22    1  0.47 2000
    3    23    1  0.47 2000
    23   24    1  0.47 2000
    3    25    1  0.47 2000
   [ system ]
    test system
    [ molecules ]
    testA 10
    testB 10
    testC 5
    """
    lines = textwrap.dedent(top_lines)
    lines = lines.splitlines()
    force_field = ForceField("test")
    topology = Topology(force_field)
    read_topology(lines=lines, topology=topology, cwdir="./")
    topology.preprocess()
    topology.volumes = {"GLY": 0.53, "GLU": 0.67, "ASP": 0.43}
    return topology

#PersistenceSpecs = namedtuple("presist", ["model", "lp", "start", "stop", "mol_idxs"])

@pytest.mark.parametrize('specs, seed, avg_step, expected', (
   # single restrain
   (
    [PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(0, 10)))],
    63594,
    [0.6533],
    [5.88, 4.57333333, 5.88, 7.84, 7.84, 9.14666667, 7.84, 3.26666667, 5.22666667, 6.53333333],
   ),
   # single restrain branched
   (
    [PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(20, 24)))],
    63594,
    [0.6533],
    [5.88, 4.57333333, 5.88, 7.84],
   ),
   # single restraint different random seed
   (
    [PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(0, 10)))],
    23893,
    [0.6533],
    [6.53333333, 4.57333333, 9.14666667, 10.45333333, 6.53333333, 7.84, 7.84, 8.49333333, 7.84, 7.84],
   ),
   # smaller range; note parser requires consecutive mol_idxs
   (
    [PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(0, 5)))],
    63594,
    [0.6533],
    [5.88, 4.57333333, 5.88, 7.84, 7.84],
   ),
   # test second group of molecules with mixed residues
   (
    [PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(10, 15)))],
    63594,
    [0.4995],
    [5.4947619, 4.49571429, 5.4947619, 6.49380952, 6.49380952],
   ),
   # test two groups of the same molecule
   (
    [PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(0, 5))),
     PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(5, 10))),],
    63594,
    [0.6533, 0.6533],
    [5.88, 4.57333333, 5.88, 7.84, 7.84, 5.88, 4.57333333, 5.88, 7.84, 7.84],
   ),
   # test three groups of the molecules two are the same one different
   (
    [PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(0, 5))),
     PersistenceSpecs(model="WCM", lp=3.0, start=0, stop=21, mol_idxs=list(range(5, 10))),
     PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(10, 15))),],
    63594,
    [0.6533, 0.6533, 0.4995],
    [5.88, 4.573, 5.88, 7.84, 7.84, 9.1467, 8.4933, 9.146, 10.4533, 10.4533,
     5.498, 4.4957, 5.4947,  6.4938, 6.4938]
   )))
def test_persistence(topology, specs, seed, avg_step, expected):
    topology.persistences = specs
    nb_engine =  NonBondEngine.from_topology(topology.molecules,
                                             topology,
                                             box=np.array([15., 15., 15.]))

    sample_end_to_end_distances(topology, nb_engine, seed=seed)
    mol_count = 0
    for batch_count, batch in enumerate(specs):
        print(batch_count)
        for mol_idx in batch.mol_idxs:
            mol = topology.molecules[mol_idx]
            mol_copy = polyply.src.meta_molecule.MetaMolecule()
            mol_copy.add_edges_from(mol.edges)
            distance = expected[mol_count]
            avg_step_length = avg_step[batch_count]

            polyply.src.restraints.set_distance_restraint(mol_copy,
                                                          batch.stop,
                                                          batch.start,
                                                          distance,
                                                          avg_step_length,
                                                          tolerance=0.0)
            for node in mol.nodes:
                print(node)
                if "distance_restraints" in mol_copy.nodes[node]:
                    restr = mol.nodes[node]["distance_restraints"]
                    ref_restr = mol_copy.nodes[node]["distance_restraints"]
                    for new, ref in zip(restr, ref_restr):
                        assert new == pytest.approx(ref, rel=1e-3)
            mol_count += 1
        batch_count += 1

def test_error(topology):
    specs = [PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(0, 10)))]
    seed = 63594
    avg_step = 0.6533
    topology.persistences = specs
    nb_engine = NonBondEngine.from_topology(topology.molecules,
                                            topology,
                                            box=np.array([1., 1., 1.]))
    with pytest.raises(IOError):
        sample_end_to_end_distances(topology, nb_engine, seed=seed)

def test_warning(caplog, topology):
    caplog.set_level(logging.WARNING)
    specs = [PersistenceSpecs(model="WCM", lp=1.5, start=0, stop=21, mol_idxs=list(range(0, 10)))]
    seed = 63594
    avg_step = 0.6533
    topology.persistences = specs
    nb_engine = NonBondEngine.from_topology(topology.molecules,
                                            topology,
                                            box=np.array([9.0, 9., 9.]))
    with caplog.at_level(logging.WARNING):
        sample_end_to_end_distances(topology, nb_engine, seed=seed)
        for record in caplog.records:
            assert record.levelname == "WARNING"
            break
        else:
            assert False
