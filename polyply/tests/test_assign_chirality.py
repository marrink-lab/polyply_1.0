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
Test assignment of chirality
"""
import pytest
import numpy as np
import networkx as nx
from polyply.src.assign_chirality import (_determine_priority,
                                         tag_chiral_centers,
                                         Chirality)

def alanine_all_atom(chirality="R"):
    """
    Molecule graph of Alanine

       H    O
       |   /
    CB-CA-C
       |   \
       NH2  O-H
    """
    molecule = nx.Graph()
    molecule.add_nodes_from(["CB", "CA", "NH2", "H", "C", "O", "OH"])
    molecule.add_edges_from([("CB", "CA"), ("CA", "H"),  ("CA", "NH2"),
                             ("CA", "C"), ("C", "O"),  ("C", "OH")])
    nx.set_node_attributes(molecule, {"CB":12, "CA":12, "NH2":17, "H":1, "C":12,
                                      "O":16, "OH":17}, "mass")
    nx.set_node_attributes(molecule, {"CB":"C", "CA":"C", "NH2":"N", "H":"H", "C":"C",
                                      "O":"O", "OH":"O"}, "atomname")

    nx.set_edge_attributes(molecule, {("CB", "CA"):1, ("CA", "H"):1,  ("CA", "NH2"):1,
                                      ("CA", "C"):1, ("C", "O"):1,  ("C", "OH"):1}, "bond_order")

    if chirality == "S":
        nx.set_node_attributes(molecule, {"CA" : np.array([-3.198, 0.036, -1.470]),
                                          "C"  : np.array([-1.798, 0.651, -1.484 ]),
                                          "O"  : np.array([-1.775, 1.784, -2.259]),
                                          "OH" : np.array([-1.382, 0.956, -0.211]),
                                          "NH2": np.array([-3.186, -1.194,  -0.698]),
                                          "CB" : np.array([-4.219,   0.992,  -0.873,]),
                                          "H"  : np.array([-3.500, -0.202, -2.512])},
                                "position")
    else:
        nx.set_node_attributes(molecule, {"CA" : np.array([3.198, 0.036, -1.470]),
                                          "C"  : np.array([1.798, 0.651,  -1.484]),
                                          "O"  : np.array([1.775, 1.784, -2.259]),
                                          "OH" : np.array([1.382, 0.956, -0.211]),
                                          "NH2": np.array([3.186, -1.194,  -0.698]),
                                          "CB" : np.array([4.219, 0.992, -0.873]),
                                          "H"  : np.array([3.500, -0.202, -2.512])},
                               "position")
    return molecule

def glucose_non_ring(chirality="D"):
    """
    Molecule graph of Glucose D-Glucose

          H1    O1
            \  //
             C1
             |
        H2 - C2 - O2
             |
 OH1 -  O3 - C3 - H3
             |
        H4 - C4 - O4-OH2
             |
        H5 - C5 - O5-OH3
             |
        H6 - C6 - O6-OH4
             |
             H7
    """
    molecule = nx.Graph()
    molecule.add_nodes_from(["H1", "H2", "H3", "H4", "H5", "H6", "H7",
                             "C1", "C2", "C3", "C4", "C5", "C6",
                             "O1", "O2", "O3", "O4", "O5", "O6"])

    molecule.add_edges_from([("H1", "C1"), ("C1", "O1"),
                             ("C1", "C2"), ("C2", "C3"),  ("C3", "C4"),
                             ("C4", "C5"), ("C5", "C6"),
                             ("H2", "C2"), ("C2", "O2"),
                             ("O3", "C3"), ("C3", "H3"), ("H4", "C4"),
                             ("C4", "O4"), ("C5", "H5"), ("C5", "O5"),
                             ("H6", "C6"), ("C6", "O6"), ("C6", "H7")])

    nx.set_edge_attributes(molecule, {("H1", "C1"):1,
                                 ("C1", "O1"):2,
                                 ("C1", "C2"):1,
                                 ("C2", "C3"):1,
                                 ("C3", "C4"):1,
                                 ("C4", "C5"):1,
                                 ("C5", "C6"):1,
                                 ("H2", "C2"):1,
                                 ("C2", "O2"):1,
                                 ("O3", "C3"):1,
                                 ("C3", "H3"):1,
                                 ("H4", "C4"):1,
                                 ("C4", "O4"):1,
                                 ("C5", "H5"):1,
                                 ("C5", "O5"):1,
                                 ("H6", "C6"):1,
                                 ("C6", "O6"):1,
                                 ("C6", "H7"):1}, "bond_order")

    nx.set_node_attributes(molecule, {"H1":1, "H2":1, "H3":1, "H4":1,
                                      "H5":1, "H6":1, "H7":1, "C1":12,
                                      "C2":12, "C3":12, "C4":12, "C5":12,
                                      "C6":12, "O1":16, "O2":17, "O3":17,
                                      "O4":17, "O5":17, "O6":17}, "mass")

    nx.set_node_attributes(molecule, {"H1":"H", "H2":"H", "H3":"H", "H4":"H",
                                      "H5":"H", "H6":"H", "H7":"H", "C1":"C",
                                      "C2":"C", "C3":"CA", "C4":"CQ", "C5":"C1",
                                      "C6":"C", "O1":"O", "O2":"O", "O3":"O",
                                      "O4":"O", "O5":"O", "O6":"O"}, "atomname")

    if chirality == "D":
        # d-glucose from 1xym; first Glucose Ligand
        nx.set_node_attributes(molecule, {"C1" : np.array([16.058, 4.592, -14.235]),
                                          "C2"  : np.array([15.925, 5.157, -12.819 ]),
                                          "C3"  : np.array([17.279, 5.785, -12.318  ]),
                                          "C4" : np.array([17.099, 6.234, -10.854]),
                                          "C5": np.array([18.280, 6.226, -9.921 ]),
                                          "C6" : np.array([17.897, 6.689, -8.506]),
                                          "O1"  : np.array([15.128, 4.051, -14.829 ]),
                                          "O2"  : np.array([14.931, 6.130, -12.824 ]),
                                          "O3"  : np.array([17.687, 6.898, -13.128 ]),
                                          "O4"  : np.array([16.804, 7.595, -10.892  ]),
                                          "O5"  : np.array([18.776, 4.921, -9.858 ]),
                                          "O6"  : np.array([16.588, 6.312,  -8.108 ]),
                                          "H1"  : np.array([17.076, 4.481, -14.581 ]),
                                          "H2"  : np.array([15.567,  4.389, -12.150 ]),
                                          "H3"  : np.array([18.071 ,  5.023, -12.359 ]),
                                          "H4"  : np.array([16.306, 5.620, -10.411 ]),
                                          "H5"  : np.array([19.010,   6.932, -10.333 ]),
                                          "H6"  : np.array([17.978, 7.780,  -8.463]),
                                          "H7"  : np.array([18.624, 6.279,  -7.787])},
                                "position")

    return molecule

def four_methyl_octane(chirality="R"):
    """
    Molecule Graph of 4 methyl Octane

                             3CH3-H4-H5-H6
                              |
                             6CH2  4C-H1-H2-H3
                               \ /
    1CH3 - 1CH2 - 2CH2 - 3CH2 - C - 4CH2 - 5CH2 - 2CH3-H7-H8-H9
    """
    molecule = nx.Graph()
    molecule.add_nodes_from(["1CH3", "2CH3", "3CH3", "4CH3",
                             "1CH2", "2CH2", "3CH2", "4CH2", "5CH2", "6CH2",
                             "C", "H1", "H2", "H3", "H4", "H5", "H6",
                             "H7", "H8", "H9"])

    molecule.add_edges_from([("1CH3", "1CH2"), ("1CH2", "2CH2"), ("2CH2", "3CH2"),
                             ("3CH2", "C"), ("C", "4CH2"),  ("C", "6CH2"),
                             ("C", "4CH3"), ("4CH2", "5CH2"), ("5CH2", "2CH3"),
                             ("6CH2", "3CH3"), ("4CH3", "H1"),("4CH3", "H2"), ("4CH3", "H3"),
                             ("3CH3", "H4"),("3CH3", "H5"), ("3CH3", "H6"),
                             ("2CH3", "H7"),("2CH3", "H8"), ("2CH3", "H9")])

    nx.set_node_attributes(molecule, {"1CH3":15, "2CH3":12, "3CH3":12, "4CH3":12,
                                      "1CH2":14, "2CH2":14, "3CH2":14, "4CH2":14,
                                      "5CH2":14, "6CH2":14, "C":12, "H1":1,"H3":1,"H2":1,
                                      "H4":1,"H5":1,"H6":1,  "H7":1,"H8":1,"H9":1}, "mass")

    nx.set_node_attributes(molecule, {"1CH3":"C", "2CH3":"C", "3CH3":"C", "4CH3":"C",
                                      "1CH2":"C", "2CH2":"C", "3CH2":"C", "4CH2":"C",
                                      "5CH2":"C", "6CH2":"C", "C"   :"C", "H1":"H","H3":"H","H2":"H",
                                      "H4":"H","H5":"H","H6":"H",  "H7":"H","H8":"H","H9":"H"}, "atomname")

    nx.set_edge_attributes(molecule, 1, "bond_order")

    return molecule

@pytest.mark.parametrize("molecule, neighbours, center, result", (
                         (alanine_all_atom(),
                          ["CB", "NH2", "H", "C"],
                          "CA",
                          {"H": 4, "C":2, "NH2": 1, "CB":3}),
                         (glucose_non_ring(),
                          ["H5", "C6", "C4", "O5"],
                          "C5",
                          {"H5": 4, "O5": 1, "C6": 3, "C4":2}),
                         (glucose_non_ring(),
                          ["H4", "C5", "C3", "O4"],
                          "C4",
                          {"H4": 4, "O4": 1, "C5": 3, "C3":2}),
                         (four_methyl_octane(),
                          ["3CH2", "6CH2", "4CH3", "4CH2"],
                          "C",
                          {"3CH2": 1, "6CH2": 3, "4CH3": 4, "4CH2":2})
                        ))
def test_find_priority(molecule, neighbours, center, result):
    status, priority_dict = _determine_priority(molecule, neighbours, center)
    assert status == True
    assert priority_dict == result


@pytest.mark.parametrize("molecule, n_centers, result", (
                         (alanine_all_atom(),
                          1,
                          {"H": 4, "C":3, "NH2": 1, "CB":2}),
                         (glucose_non_ring(),
                          4,
                          {"H5": 4, "O4": 1, "C6": 3, "C4":2}),
                        (four_methyl_octane(),
                         1,
                         {"3CH2": 1, "6CH2": 3, "4CH3": 4, "4CH2":2})
                        ))
def test_tag_chiral_centers(molecule, n_centers, result):
    centers = tag_chiral_centers(molecule, center_atom="C")
    assert len(centers) == n_centers
    assert len(nx.get_node_attributes(molecule, "chiral_id")) == n_centers

@pytest.mark.parametrize("molecule, chiralities", (
                         (alanine_all_atom("R"),
                          {"CA": "R"}),
                         (alanine_all_atom("S"),
                          {"CA": "S"}),
                         (glucose_non_ring("D"),
                          {"C2":"R", "C3":"S", "C4":"R","C5":"R"}),
                        ))
def test_assign_chirality(molecule, chiralities):
    Chirality()._assign_chirality(molecule)
    for center, chirality in chiralities.items():
        assert molecule.nodes[center]["chirality"] == chirality
