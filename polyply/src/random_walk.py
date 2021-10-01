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

import random
import numpy as np
from numpy.linalg import norm
import networkx as nx
from .processor import Processor
from .linalg_functions import norm_sphere
from .linalg_functions import _vector_angle_degrees
from .graph_utils import neighborhood
from .meta_molecule import _find_starting_node
"""
Processor implementing a random-walk to generate
coordinates for a meta-molecule.
"""


def pbc_complete(point, maxdim):
    """
    Wrap point around pbc conditions to keep
    points from being larger than the compontents of
    the maxdim vector. Note that all coodinates in
    polyply are definiet positive.

    Parameters:
    -----------
    point: np.ndarray
    maxdim: np.ndarray

    Returns:
    --------
    np.ndarray
    """
    return point % maxdim


def _take_step(vectors, step_length, coord, box):
    """
    Given a list of unit `vectors` choose one randomly,
    multiply it by the `step_length` and add to `coord`.

    Parameters
    ----------
    vectors: list[np.ndarray(n, 3)]
    step_length: float
    coord: np.ndarray(3)

    Returns
    -------
    np.ndarray(3)
        the new coordinate
    int
        the index of the vector choosen
    """
    index = random.randint(0, len(vectors) - 1)
    new_coord = coord + vectors[index] * step_length
    new_coord = pbc_complete(new_coord, box)
    return new_coord, index


def not_exceeds_max_dimensions(point, maxdim):
    """
    Check if point is within the compontents of
    the maxdim vector. Note that all coodinates in
    polyply are definiet positive.

    Parameters:
    -----------
    point: np.ndarray
    maxdim: np.ndarray

    Returns:
    --------
    bool
    """
    return np.all(point < maxdim) and np.all(point > np.array([0., 0., 0.]))

def is_restricted(point, old_point, node_dict):
    """
    The function checks, if the step `old_point` to
    `point` is in a direction as defined by a plane
    with normal and angle as set in the `node_dict`
    options with keyword rw_options. It is true
    if the direction vector point-old_point is
    pointing some max angle from the normal of the plane
    in the same direction as an angle specifies.
    To check we compute the signed angle of the vector
    `point`-`old_point` has with the plane defined
    by a normal and compare to the reference angle.
    The angle needs to have the same sign as angle
    and not be smaller in magnitude.

    Parameters:
    -----------
    point: np.ndarray
    old_point: np.ndarray
    node_dict:  dict["rw_options"][[np.ndarray, float]]
        dict with key rw_options containing a list of lists
        specifying a normal and an angle

    Returns:
    -------
    bool
    """

    if not "rw_options" in node_dict:
        return True

    normal, ref_angle = node_dict["rw_options"][0]
    # check condition 1
    sign = np.sign(np.dot(normal, point - old_point))
    if sign != np.sign(ref_angle):
        return False

    # check condition 2
    angle = _vector_angle_degrees(normal, point - old_point)
    if angle > np.abs(ref_angle):
        return False
    return True


def in_cylinder(point, parameters):
    """
    Assert if a point is within a cylinder or outside a
    cylinder as defined in paramters. Note the cylinder
    is z-aligned always.

    Parameters:
    -----------
    point: np.ndarray
        reference point
    parameters: abc.iteratable

    Returns:
    --------
    bool
    """
    in_out = parameters[0]
    diff = parameters[1] - point
    radius = norm(diff[:2])
    half_heigth = diff[2]
    if in_out == "in" and radius < parameters[2] and np.abs(half_heigth) < parameters[3]:
        return True
    elif in_out == "out" and (radius > parameters[2] or half_heigth > np.abs(parameters[3])):
        return True
    else:
        return False

def in_rectangle(point, parameters):
    """
    Assert if a point is within a rectangle or outside a
    rectangle as defined in paramters:

    Parameters:
    -----------
    point: np.ndarray
        reference point
    parameters: abc.iteratable

    Returns:
    --------
    bool
    """
    in_out = parameters[0]
    diff = parameters[1] - point
    check = [np.abs(dim) < max_dim for dim,
             max_dim in zip(diff, parameters[2:])]
    if in_out == "in" and not all(check):
        return False
    elif in_out == "out" and all(check):
        return False
    else:
        return True


def in_sphere(point, parameters):
    """
    Assert if a point is within a sphere or outside a
    sphere as defined in paramters:

    Parameters:
    -----------
    point: np.ndarray
        reference point
    parameters: abc.iteratable

    Returns:
    --------
    bool
    """
    in_out = parameters[0]
    r = norm(parameters[1] - point)
    if in_out == 'in' and r > parameters[2]:
        return False
    elif in_out == 'out' and r < parameters[2]:
        return False
    else:
        return True


# methods of geometrical restraints known to polyply
RESTRAINT_METHODS = {"cylinder": in_cylinder,
                     "rectangle": in_rectangle,
                     "sphere":  in_sphere}


def fulfill_geometrical_constraints(point, node_dict):
    """
    Assert if a point fulfills a geometrical constraint
    as defined by the "restraint" key in a dictionary.
    If there is no key "restraint" the function returns true.

    Parameters:
    -----------
    point: np.ndarray
        reference point

    node_dict: :class:ditc

    Returns:
    --------
    bool
    """
    if "restraints" not in node_dict:
        return True

    for restraint in node_dict['restraints']:
        restr_type = restraint[-1]
        if not RESTRAINT_METHODS[restr_type](point, restraint):
            return False

    return True


def _find_starting_node(meta_molecule):
    """
    Find the first node that has coordinates if there is
    otherwise return first node in list of nodes.
    """
    for node in meta_molecule.nodes:
        if "build" not in meta_molecule.nodes[node]:
            return node
    return next(iter(meta_molecule.nodes()))


class RandomWalk(Processor):
    """
    Add coordinates at the meta_molecule level
    through a random walk for all nodes which have
    build defined as true.
    """
    def __init__(self,
                 mol_idx,
                 nonbond_matrix,
                 start=np.array([0, 0, 0]),
                 step_fudge=0.8,
                 maxiter=80,
                 maxdim=None,
                 max_force=1e3,
                 vector_sphere=norm_sphere(5000),
                 start_node=None,
                 nrewind=5):

        self.mol_idx = mol_idx
        self.nonbond_matrix = nonbond_matrix
        self.start = start
        self.maxiter = maxiter
        self.maxdim = maxdim
        self.vector_sphere = vector_sphere
        self.success = False
        self.max_force = max_force
        self.step_fudge = step_fudge
        self.start_node = start_node
        self.nrewind = nrewind
        self.placed_nodes = []

    def _rewind(self, current_step):
        nodes = [node for _, node in self.placed_nodes[-self.nrewind:-1]]
        self.nonbond_matrix.remove_positions(self.mol_idx, nodes)
        step_count = self.placed_nodes[-self.nrewind][0]
        self.placed_nodes = self.placed_nodes[:-self.nrewind]
        return step_count

    def _is_overlap(self, point, node, nrexcl=1):
        neighbours = neighborhood(self.molecule, node, nrexcl)
        force_vect = self.nonbond_matrix.compute_force_point(point,
                                                             self.mol_idx,
                                                             node,
                                                             neighbours,
                                                             potential="LJ")
        return norm(force_vect) > self.max_force


    def checks_milestones(self, current_node, current_position, fudge=0.7):

        if 'distance_restraints' in self.molecule.nodes[current_node]:
            for restraint in self.molecule.nodes[current_node]['distance_restraints']:
                ref_node, upper_bound, lower_bound = restraint
                ref_pos = self.nonbond_matrix.get_point(self.mol_idx, ref_node)
                current_distance = self.nonbond_matrix.pbc_min_dist(current_position, ref_pos)
                if current_distance > upper_bound:
                    return False

                if current_distance < lower_bound:
                    return False

        return True

    def update_positions(self, vector_bundle, current_node, prev_node):
        """
        Take an array of unit vectors `vector_bundle` and generate the coordinates
        for `current_node` by adding a random vector to the position of the previous
        node `prev_node`. The length of that vector is defined as 2 times the vdw-radius
        of the two nodes. The position is updated in place.

        Parameters
        ----------
        vector_bunde: np.ndarray(m,3)
        meta_molecule: :class:`polyply.src.meta_molecule.MetaMolecule`
        current_node: node_key[int, str]
        prev_node: node_key[int, str]
        topology: :class:`polyply.src.topology.Topology`
        maxiter: int
           maximum number of iterations
        """
        last_point = self.nonbond_matrix.get_point(self.mol_idx, prev_node)
        step_length = self.step_fudge * self.nonbond_matrix.get_interaction(self.mol_idx,
                                                                            self.mol_idx,
                                                                            prev_node,
                                                                            current_node)[0]
        step_count = 0
        while True:

            new_point, index = _take_step(vector_bundle, step_length, last_point, self.maxdim)
            if fulfill_geometrical_constraints(new_point, self.molecule.nodes[current_node])\
                and self.checks_milestones(current_node, new_point, step_length)\
                and is_restricted(new_point, last_point, self.molecule.nodes[current_node])\
                and not self._is_overlap(new_point, current_node):

                    self.nonbond_matrix.add_positions(new_point,
                                                      self.mol_idx,
                                                      current_node,
                                                      start=False)
                    return True

            elif step_count == self.maxiter:
                return False

            else:
                step_count += 1
                vector_bundle = np.delete(vector_bundle, index, axis=0)

    def _random_walk(self, meta_molecule):
        """
        Perform a random_walk to build positions for a meta_molecule, if
        no position is present for an atom.

        Parameters
        ----------
        meta_molecule:  :class:`polyply.src.meta_molecule.MetaMolecule`
        """
        if not self.start_node:
            first_node = _find_starting_node(meta_molecule)
        else:
            first_node = self.start_node

        meta_molecule.root = first_node

        if "position" not in meta_molecule.nodes[first_node]:
            constrained = fulfill_geometrical_constraints(self.start,
                                                          self.molecule.nodes[first_node])

            if constrained and not self._is_overlap(self.start, first_node):
                self.nonbond_matrix.add_positions(self.start,
                                                  self.mol_idx,
                                                  first_node,
                                                  start=True)
                self.success = True
            else:
                self.success = False
                return

        vector_bundle = self.vector_sphere.copy()
        count = 0
        path = list(meta_molecule.search_tree.edges)
        step_count = 0

        while step_count < len(path):
            prev_node, current_node = path[step_count]

            if not meta_molecule.nodes[current_node]["build"]:
                step_count += 1
                continue

            status = self.update_positions(vector_bundle,
                                           current_node,
                                           prev_node)
            self.success = status
            self.placed_nodes.append((step_count, current_node))

            # in principle this count here is not needed, however,
            # we need to check the performance in terms of strucutre
            # generation before doing any adjustments here
            if not self.success and count < self.maxiter:
                if len(self.placed_nodes) < self.nrewind+1:
                    return
                step_count = self._rewind(step_count)
            elif not self.success:
                return
            else:
                count = 0
                step_count += 1

            count += 1

    def run_molecule(self, meta_molecule):
        """
        Perform the random walk for a single molecule.
        """
        self.molecule = meta_molecule
        self._random_walk(meta_molecule)
        return meta_molecule
