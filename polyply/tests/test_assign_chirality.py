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
import networkx as nx
from polyply.src.assign_chirality import (_determine_priority,
                                         tag_chiral_centers,
                                         Chirality)

def alanine_all_atom():
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
    nx.set_node_attributes(molecule, {"CB":15, "CA":12, "NH2":17, "H":1, "C":12,
                                      "O":16, "OH":17}, "mass")
    nx.set_node_attributes(molecule, {"CB":"C", "CA":"C", "NH2":"N", "H":"H", "C":"C",
                                      "O":"O", "OH":"O"}, "atomname")
    nx.set_node_attributes(molecule, {CA :np.array([ -6.578,  0.552,  0.043]),
                                      C  :np.array([ -5.246,  1.181, -0.264]),
                                      O  :np.array([ -5.201,  2.319, -0.703]),
                                      OH :np.array([ -4.097,  0.506, -0.054]),
                                      N  :np.array([ -7.380,  0.494, -1.167]),
                                      CB :np.array([ -6.424, -0.854,  0.600]),
                                      H  :np.array([ -7.106,  1.171,  0.798])})

   return molecule

def alanine_united_atom():
    """
    Molecule graph of Alanine

            O
           /
    CB-CA-C
       |   \
       NH2  O-H
    """
    molecule = nx.Graph()
    molecule.add_nodes_from(["CB", "CA", "NH2", "C", "O", "OH"])
    molecule.add_edges_from([("CB", "CA"), ("CA", "NH2"),
                             ("CA", "C"), ("C", "O"),  ("C", "OH")])
    nx.set_node_attributes(molecule, {"CB":15, "CA":12, "NH2":17, "H":1, "C":12,
                                      "O":16, "OH":17}, "mass")
    return molecule

def glucose_non_ring():
    """
    Molecule graph of Glucose

          H1    O1
            \  /
             C1
             |
        H2 - C2 - OH1
             |
       2HO - C3 - H3
             |
        H4 - C4 - O3-H3
             |
        H5 - C5 - O4-H4
             |
        H6 - C6 - O5-H5
             |
             H
    """
    molecule = nx.Graph()
    molecule.add_nodes_from(["H1", "H2", "H3", "H4", "H5", "H6", "H7",
                             "C1", "C2", "C3", "C4", "C5", "C6",
                             "O1", "OH1", "OH3", "OH4", "OH5"])

    molecule.add_edges_from([("H1", "C1"), ("C1", "O1"),
                             ("C1", "C2"), ("C2", "C3"),  ("C3", "C4"),
                             ("C4", "C5"),
                             ("C5", "C6"), ("H2", "C2"), ("C2", "OH1"),
                             ("OH2", "C3"), ("C3", "H3"), ("H4", "C4"),
                             ("C4", "OH3"), ("C5", "H5"), ("C5", "OH4"),
                             ("H6", "C6"), ("C6", "OH5"), ("C6", "H7")])

    nx.set_node_attributes(molecule, {"H1":1, "H2":1, "H3":1, "H4":1,
                                      "H5":1, "H6":1, "H7":1, "C1":12,
                                      "C2":12, "C3":12, "C4":12, "C5":12,
                                      "C6":12, "O1":16, "OH1":17, "OH3":17,
                                      "OH4":17, "OH5":17, "OH2":17}, "mass")

    nx.set_node_attributes(molecule, {"H1":"H", "H2":"H", "H3":"H", "H4":"H",
                                      "H5":"H", "H6":"H", "H7":"H", "C1":"C",
                                      "C2":"C", "C3":"CA", "C4":"CQ", "C5":"C1",
                                      "C6":"C", "O1":"O", "OH1":"O", "OH3":"O",
                                      "OH4":"O", "OH5":"O"}, "atomname")

    chiral_graph = nx.Graph()
    return molecule

def four_methyl_octane():
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
    return molecule

@pytest.mark.parametrize("molecule, neighbours, center, result", (
                         (alanine_all_atom(),
                          ["CB", "NH2", "H", "C"],
                          "CA",
                          {"H": 4, "C":3, "NH2": 1, "CB":2}),
                         (glucose_non_ring(),
                          ["H5", "C6", "C4", "OH4"],
                          "C5",
                          {"H5": 4, "OH4": 1, "C6": 3, "C4":2}),
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
                          {"H5": 4, "OH4": 1, "C6": 3, "C4":2}),
                         (four_methyl_octane(),
                          1,
                          {"3CH2": 1, "6CH2": 3, "4CH3": 4, "4CH2":2})
                        ))
def test_tag_chiral_centers(molecule, n_centers, result):
    centers = tag_chiral_centers(molecule, center_atom="C")
    assert len(centers) == n_centers
    assert len(nx.get_node_attributes(molecule, "chiral_id")) == n_centers


@pytest.mark.parametrize("molecule chiralities", (
                         (alanine_all_atom(),
                          {"CA": "R"},
                     #    (glucose_non_ring(),
                     #     {"H5": 4, "OH4": 1, "C6": 3, "C4":2}),
                         (four_methyl_octane(),
                          "C",
                          {"3CH2": 1, "6CH2": 3, "4CH3": 4, "4CH2":2})
                        ))
def test_assign_chirality(molecule, chiralities):
    molecule = Chirality.assign_chirality(molecule)
