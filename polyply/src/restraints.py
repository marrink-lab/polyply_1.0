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
import networkx as nx
from .graph_utils import compute_avg_step_length

def set_distance_restraint(molecule, target_node, ref_node, distance, avg_step_length):
    """
    Given that a target_node is supposed to be restrained to a reference node at a given
    distance, this function computes for each node in the molecule an upper and lower
    bound references distance that this node needs to have in relation to the target node.
    Those two distances are stored in the 'restraint' attribute of the node.  This information
    is picked up by the `:func:polyply.src.random_walk.checks_milstones` function, which
    then checks that each bound is met by a step in the random_walk.

    The upper_bound is defined as the graph distance of a given node from the target_node
    times the average step_length plus the cartesian distance at which the nodes are
    restrained.

    The lower_bound is defined as the average step-length multiplied by the distance of the
    node from the reference node.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
    target_node: collections.abc.hashable
        node key of the node that is supposed to be at a
        distance from ref_node
    ref_node: collections.abc.hashable
        node key if this is a distance restraint
    distance: float
        the cartesian distance between nodes in nm
    nonbond_matrix: `polyply.src.nonbond_matrix.NonBondMatrix`
    """
    graph_distances_target = nx.single_source_shortest_path_length(molecule,
                                                                   source=target_node,
                                                                   cutoff=None)

    graph_distances_ref = nx.single_source_shortest_path_length(molecule,
                                                                source=ref_node,
                                                                cutoff=None)
    for node in molecule.nodes:
        if node == target_node:
            graph_distance = 1.0
        else:
            graph_distance = graph_distances_target[node]

        upper_bound = graph_distance * avg_step_length + distance

        avg_needed_step_length = distance / graph_distances_target[ref_node]
        lower_bound = avg_needed_step_length * graph_distances_ref[node]

        current_restraints = molecule.nodes[node].get('distance_restraints', [])
        molecule.nodes[node]['distance_restraints'] = current_restraints + [(ref_node, upper_bound, lower_bound)]

def set_restraints(topology, nonbond_matrix):
    """
    Set position and/or distance restraints for all molecules.
    """
    for mol_name, mol_idx in topology.distance_restraints:
        distance_restraints = topology.distance_restraints[(mol_name, mol_idx)]
        mol = topology.molecules[mol_idx]
        for ref_node, target_node in distance_restraints:
            avg_step_length, total = compute_avg_step_length(mol,
                                                             mol_idx,
                                                             target_node,
                                                             nonbond_matrix,
                                                             stop_node=ref_node)

            distance = distance_restraints[(ref_node, target_node)]
            set_distance_restraint(mol, target_node, ref_node, distance, avg_step_length)
