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
Conditionals for all polyply builders.
"""
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


def fulfill_geometrical_constraints(nobond_matrix, molecule, mol_idx, current_node, current_position):
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
    node_dict = molecule.nodes[current_node]
    if "restraints" not in node_dict:
        return True

    for restraint in node_dict['restraints']:
        restr_type = restraint[-1]
        if not RESTRAINT_METHODS[restr_type](point, restraint):
            return False

    return True


def not_is_overlap(nobond_matrix, molecule, mol_idx, current_node, current_position):
    neighbours = neighborhood(molecule, current_node, molecule.nrexcl)
    force_vect = nonbond_matrix.compute_force_point(current_position,
                                                    mol_idx,
                                                    node,
                                                    neighbours,
                                                    potential="LJ")
    return norm(force_vect) < self.max_force


def checks_milestones(nonbond_matrix, molecule, mol_idx, current_node, current_position):

    if 'distance_restraints' in molecule.nodes[current_node]:
       for restraint in molecule.nodes[current_node]['distance_restraints']:
           ref_node, upper_bound, lower_bound = restraint
           ref_pos = nonbond_matrix.get_point(mol_idx, ref_node)
           current_distance = nonbond_matrix.pbc_min_dist(current_position, ref_pos)
           if current_distance > upper_bound:
               return False

           if current_distance < lower_bound:
               return False

        return True

def is_restricted(nonbond_matrix, molecule, mol_idx, current_node, current_position):
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
    node_dict = molecule.nodes[current_node]
    if not "rw_options" in node_dict:
        return True

    # find the previous node
    prev_node = molecule.predecessors(current_node)
    prev_position = molecule.nodes[prev_node]["position"]

    normal, ref_angle = node_dict["rw_options"][0]
    # check condition 1
    sign = np.sign(np.dot(normal, current_position - prev_position))
    if sign != np.sign(ref_angle):
        return False

    # check condition 2
    angle = _vector_angle_degrees(normal, current_position - prev_position)
    if angle > np.abs(ref_angle):
        return False
    return True
