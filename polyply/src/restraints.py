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
from .graph_utils import compute_avg_step_length, get_all_predecessors

def set_distance_restraint(molecule,
                           target_node,
                           ref_node,
                           distance,
                           avg_step_length,
                           tolerance):
    """
    Given that a target_node is supposed to be restrained to a reference node at a given
    distance, this function computes for each node in the molecule an upper and lower
    bound references distance that this node needs to have in relation to the target node.
    Those two distances are stored in the 'restraint' attribute of the node.  This information
    is picked up by the `:func:polyply.src.random_walk.checks_milestones` function, which
    then checks that each bound is met by a step in the random_walk.

    The upper_bound is defined as the graph distance of a given node from the target_node
    times the avg_step_length plus the cartesian distance at which the node is
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
    avg_step_length: float
        average step length (nm)
    tolerance: float
        absolute tolerance (nm)
    """
    # if the target node is a predecssor to the ref node the order
    # needs to be reveresed because the target node will be placed
    # before the ref node
    # we get the path lying between target and reference node
    ancestor = nx.algorithms.lowest_common_ancestor(molecule.search_tree, target_node, ref_node)
    if ancestor == target_node:
        ref_node, target_node = target_node, ref_node
    elif ancestor != ref_node:
        msg=("Your distance restraint between node { } { } is not valid. "
             "Likely you are trying to apply distance restraints on a "
             "branched molecule. This is not fully supported yet. ")
        raise OSError(msg.format(ref_node, target_node))

    path = get_all_predecessors(molecule.search_tree, node=target_node, start_node=ref_node)

    # graph distances can be computed from the path positions
    graph_distances_ref = {node: path.index(node) for node in path}
    graph_distances_target = {node: len(path) - 1 - graph_distances_ref[node] for node in path}

    for node in path:
        if node == target_node:
            graph_distance = 1.0
        elif node == ref_node:
            continue
        else:
            graph_distance = graph_distances_target[node]

        upper_bound = graph_distance * avg_step_length + distance + tolerance

        avg_needed_step_length = distance / graph_distances_target[ref_node]
        lower_bound = avg_needed_step_length * graph_distances_ref[node] - tolerance

        current_restraints = molecule.nodes[node].get('distance_restraints', [])
        molecule.nodes[node]['distance_restraints'] = current_restraints + [(ref_node,
                                                                             upper_bound,
                                                                             lower_bound)]

def set_restraints(topology, nonbond_matrix):
    """
    Set position and/or distance restraints for all molecules.

    topology: `:class:polyply.src.topology.nonbond_matrix`
        the topology of the system
    nonbond_matrix: `:class:polyply.src.NonBondEngine`
        the nonbond matrix which stores all the pairwise interaction
        parameters and positions
    """
    for mol_name, mol_idx in topology.distance_restraints:
        distance_restraints = topology.distance_restraints[(mol_name, mol_idx)]
        mol = topology.molecules[mol_idx]

        for ref_node, target_node in distance_restraints:
            path = list(mol.search_tree.edges)
            avg_step_length, _ = compute_avg_step_length(mol,
                                                         mol_idx,
                                                         nonbond_matrix,
                                                         path)

            distance, tolerance = distance_restraints[(ref_node, target_node)]
            set_distance_restraint(mol, target_node, ref_node, distance, avg_step_length, tolerance)
