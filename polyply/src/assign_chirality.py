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
import numpy as np
import networkx as nx
from .processor import Processor
from .apply_links import neighborhood
from .linalg_functions import signed_angle, projection
from .errors import MaxIterationError

def which_chirality(center, subsituents):
    """
    Determine the chirality of center with four
    substituents. Note the `atoms` need to be a list
    of tuples with the first entry the priority.

    Paramteres:
    -----------
    center: np.ndarray
    subsituents: list(tuple(int, np.ndarray))

    Returns:
    -----------
    float
       sin of singed angle
    """
    subsituents.sort()
    a, b, c, origin_proj, normal = projection(center,
                                              subsituents[0][1], subsituents[1][1],
                                              subsituents[2][1], subsituents[3][1])
    a_zero = a - origin_proj
    bc = b - c
    ang = signed_angle(a_zero, bc, normal)

    if np.sin(ang) > 0:
        return "R"
    else:
        return "S"


def _unique_masses(masses):
    mass_list = [mass[0] for mass in masses]
    unique = [item for item, count in collections.Counter(
        mass_list).items() if count < 2]
    return unique


def _determine_priority(molecule, neighbours, center, max_depth=10):
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
    max_depth:
       maximum number of edges for neighbour search

    Returns
    -------
    dict
    """
    masses = [(molecule.nodes[node]['mass'], node) for node in neighbours]
    priority = {}
    degree = 1
    priorities = [4, 3, 2, 1]
    while True:

        if degree == max_depth:
            msg = ("When determining priority a tie is "
                   "remains unresolved after looking at "
                   "the {} degree neighbours. Increase "
                   "depth of search if you are sure the "
                   "molecule is chiral.")
            raise MaxIterationError(msg.format(degree))

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
            neighbours, paths = neighborhood(molecule, node,
                                             degree, min_length=degree+1,
                                             not_cross=[center], return_paths=True)
            for neigh in neighbours:
                if degree == 1:
                    edge = (node, neigh)
                else:
                    edge = (neigh, paths[neigh][-2])

                bond_order = molecule.edges[edge]["bond_order"]
                mass += molecule.nodes[neigh]["mass"] * bond_order

            masses[idx] = (mass, node)
            idx += 1
        degree += 1
        if masses and all([mass[0] == 0 for mass in masses]):
            status = False
            break

    return status, priority


def tag_chiral_centers(molecule, center_atom="C", priorities={}):
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
    centers = []
    for node, degree in molecule.degree:
        if degree == 4 and molecule.nodes[node]["atomname"].startswith(center_atom):
            neighbours = molecule.neighbors(node)

            if node in priorities:
                is_chiral = True
                priority = priorities[node]
            else:
                print("---")
                is_chiral, priority = _determine_priority(
                    molecule, neighbours, node)
                if not is_chiral:
                    continue

            molecule.nodes[node]["chiral_id"] = idx
            molecule.nodes[node]["priority"] = priority
            molecule.nodes[node]["chiral"] = True
            idx += 1
            centers.append(node)

    return centers


class Chirality(Processor):
    """
    For each chiral center of an atomistic
    molecule assign the chirality.
    """

    def __init__(self, priorities={}):
        self.priorities = priorities

    def _assign_chirality(self, molecule):
        """
        """
        centers = tag_chiral_centers(molecule,
                                     center_atom="C",
                                     priorities=self.priorities)
        for center in centers:
            priority = molecule.nodes[center]["priority"]
            positions = []
            for node, priority_idx in priority.items():
                positions.append((priority_idx, molecule.nodes[node]["position"]))

            chirality = which_chirality(molecule.nodes[center]["position"], positions)
            molecule.nodes[center]["chirality"] = chirality

    def run_molecule(self, meta_molecule):
        """
        """
        self._assign_chirality(meta_molecule.molecule)
        return meta_molecule
