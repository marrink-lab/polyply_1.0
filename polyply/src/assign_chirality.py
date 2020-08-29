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
High level API for the polyply tool to analyze tacticity
"""
import collections
import networkx as nx
from vermouth.graph_utils import make_residue_graph
from .processor import Processor
from .apply_links import neighborhood
from .linalg_functions import which_chirality


def _unique_masses(masses):
    mass_list = [ mass[0] for mass in masses]
    unique = [item for item, count in collections.Counter(mass_list).items() if count < 2]
    return unique

def _determine_priority(molecule, neighbours, center):
    """
    Given a chiral `center` in a `molecule` and the `neighbours`
    of that center, determine the priority according to CIP
    defenition for chiral centers.

    Parameters:
    ------------
    molecule: vermouth.molecule
    neighbours: list
        needs to be exactly four neighbours
    center:
       node key of the center

    Returns
    -------
    dict
    """
    masses = [(molecule.nodes[node]['mass'], node) for node in neighbours]
    priority = {}
    degree = 1
    priorities=[4,3,2,1]
    max_count=10
    while True:

        if degree == 10:
           raise Error

        masses.sort()
        unique = _unique_masses(masses)
        idx = 0
        remove = []
        for mass, node in masses:
            if mass in unique:
                mass, node = masses[idx]
                priority[node] = priorities[idx]
                remove.append(((mass, node), priorities[idx]))
            idx += 1

        if all([key in priority for key in priorities]):
           status = True
           break
        else:
           for entry in remove:
               masses.remove(entry[0])
               priorities.remove(entry[1])

        idx = 0
        for mass, node in masses:
            mass = 0
            neighbours = neighborhood(molecule, node,
                                      degree, min_length=degree,
                                      not_cross=[center])
            for neigh in neighbours:
                mass += molecule.nodes[neigh]["mass"]
            masses[idx] = (mass, node)
            idx += 1
        degree += 1
        if masses and all([mass[0] == 0 for mass in masses]):
           status = False
           break

    return status, priority


def tag_chiral_centers(molecule, center_atom="C", priorities=None):
    """
    Given a `molecule` identify all chiral centers
    and determine the priority of the subsituents
    according the CIP rules. Centers are defined
    as nodes of degree 4 and having the atomname
    defined in `center_atom`.

    Parameters:
    ------------
    molecule: vermouth.molecule
    center_atom: string

    Returns:
    -------
    nx.Graph
    """
    idx = 0
    nx.set_node_attributes(molecule, False, "chiral")
    centers=[]
    for node, degree in molecule.degree:
        if degree == 4 and molecule.nodes[node]["atomname"].startswith(center_atom):
            neighbours = molecule.neighbors(node)

            if node in priorities:
               is_chiral = True
               priority = priorities[nodr]
            else:
               is_chiral, priority = _determine_priority(molecule, neighbours, node)
               if not is_chiral:
                  continue

            nx.set_node_attributes(molecule, {node:idx}, "chiral_id")
            nx.set_node_attributes(molecule, {node:priority}, "priority")
            nx.set_node_attributes(molecule, {node:True}, "chiral")
            idx += 1
            centers.append(node)
    return centers

class Chirality(Processor):
    """
    For each chiral center of an atomistic
    molecule assign the chirality.
    """

    def __init__(priorities):
        self.priorities = priorities

    @staticmethod
    def _assign_chirality(molecule):
        """
        """
        centers = tag_chiral_centers(molecule.molecule, center_atom="C")

        for center in centers:
            priority = molecule[center]["priority"]
            positions = [(priority_idx,
                          molecule.nodes[node]["position"]) for node, priority_idx in priority]
            positions.sort()
            chirality = which_chirality(positions)
            molecule[center]["chirality"] = chirality

    def run_molecule(self, meta_molecule):
        """
        """
        self._assign_chirality(meta_molecule.molecule)
        return meta_molecule
