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
from polyply.src.graph_utils import compute_avg_step_length, get_all_predecessors

def upper_bound(avg_step_length, distance, graph_dist, tolerance=0):
    bound = graph_dist * avg_step_length + distance + tolerance
    return bound


def lower_bound(distance, graph_dist_ref, graph_dist_target, tolerance=0):
    avg_needed_step_length = distance / graph_dist_target
    bound = avg_needed_step_length * graph_dist_ref - tolerance
    return bound


def _set_restraint_on_path(graph,
                           path,
                           avg_step_length,
                           distance,
                           tolerance,
                           ref_node=None,
                           target_node=None):

    # graph distances can be computed from the path positions
    graph_distances_ref = {node: path.index(node) for node in path}
    graph_distances_target = {node: len(path) - 1 - graph_distances_ref[node] for node in path}

    if not target_node:
        target_node = path[-1]

    if not ref_node:
        ref_node = path[0]

    for node in path[1:]:
        if node == target_node:
            graph_dist_ref = 1.0
        else:
            graph_dist_ref = graph_distances_target[node]

        graph_dist_target = graph_distances_target[node]
        up_bound = upper_bound(avg_step_length, distance, tolerance, graph_dist_ref)
        low_bound = lower_bound(distance, graph_dist_ref, graph_dist_target, tolerance)

        current_restraints = graph.nodes[node].get('distance_restraints', [])
        graph.nodes[node]['distance_restraints'] = current_restraints + [(ref_node,
                                                                          up_bound,
                                                                          low_bound)]


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
    # First we need to figure out if the two nodes to be restrained are
    # each others common ancestor. This breaks on cyclic graphs
    ancestor = nx.find_lowest_common_ancestor(
        molecule.search_tree, ref_node, target_node)
    # if the ancestor is equal to the target node we have to switch the
    # reference and target node
    paths = []
    if ancestor == target_node:
        ref_nodes = [target_node]
        target_nodes = [ref_node]
        distances = [distance]
    # if ancestor is neither target nor ref_node, there is no continues path
    # between the two. In this case we have to split the distance restraint
    # into two parts
    elif ancestor != ref_node:
        # !!!!!!!!! THIS NEEDS TO HAVE THE APPROPIATE ORDER !!!!!!!!!!!!!!!
        ref_nodes = [ancestor, ref_node]
        target_nodes = [ref_node, target_node]

        # the first part is the average expected distance between the target node and the
        # common ancestor
        path = get_all_predecessors(molecule.search_tree,
                                    node=target_node,
                                    start_node=ancestor)

        graph_dist_ref_target = len(path) - 1
        up_bound = upper_bound(avg_step_length, distance, tolerance, graph_dist_ref_target)
        low_bound = lower_bound(distance, graph_dist_ref_target, graph_dist_ref_target, tolerance)
        partial_distance = (up_bound + low_bound) / 2.
        paths.append(path)
        distances = [partial_distance]
        # then we put the other distance restraint that acts like the full one but
        # on a partial path between ancestor and target node
        path = get_all_predecessors(molecule.search_tree,
                                    node=target_node,
                                    start_node=ref_node)
        paths.append(path)
        distances.append(distance)
    # if ancestor is equal to ref node all order is full-filled and we just proceed
    else:
        ref_nodes = [ref_node]
        target_nodes = [target_node]
        distances = [distance]

    if not paths:
        paths = [get_all_predecessors(molecule.search_tree,
                                      node=target_node,
                                      start_node=ref_node)]

    for ref_node, target_node, dist, path in zip(ref_nodes, target_nodes, distances, paths):
        _set_restraint_on_path(molecule,
                               path,
                               avg_step_length,
                               dist,
                               tolerance,
                               ref_node,
                               target_node)

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

        if set(dict(nx.degree(mol)).values()) != {1, 2}:
            raise IOError(
                "Distance restraints currently can only be applied to linear molecules.")

        for ref_node, target_node in distance_restraints:
            path = list(mol.search_tree.edges)
            avg_step_length, _ = compute_avg_step_length(mol,
                                                         mol_idx,
                                                         nonbond_matrix,
                                                         path)

            distance, tolerance = distance_restraints[(ref_node, target_node)]
            set_distance_restraint(
                mol, target_node, ref_node, distance, avg_step_length, tolerance)
