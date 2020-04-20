import random
import networkx as nx
import numpy as np
import polyply
from polyply.src.processor import Processor


def _take_step(vectors, step_length, coord):
    index = random.randint(0, len(vectors) - 1)
    new_coord = coord + vectors[index] * step_length
    return new_coord, index


def _is_overlap(meta_molecule, new_point, tol, fudge=1):

    for node in meta_molecule:
        try:
            coord = meta_molecule[node]["position"]
        except KeyError:
            continue

        if np.linalg.norm(coord - new_point) < tol * fudge:
           return True

    return False


def _combination(radius_A, radius_B):
    return (radius_A + radius_B) / 2.


def update_positions(vector_bundel, meta_molecule, current_node, prev_node):
    current_vectors = np.zeros(vector_bundel.shape)
    current_vectors[:] = vector_bundel[:]
    last_point = meta_molecule.nodes[prev_node]["position"]
    resname = meta_molecule.nodes[prev_node]["resname"]

    # add combination rule
    vdw_radius = meta_molecule.volumes[resname]
    step_length = 2*vdw_radius

    while True:
        new_point, index = _take_step(vector_bundel, step_length, last_point)

        if not _is_overlap(meta_molecule, new_point, tol=vdw_radius):
            meta_molecule.nodes[current_node]["position"] = new_point
            break
        else:
            vector_bundel = np.delete(vector_bundel, index, axis=0)


class RandomWalk(Processor):
    """

    """

    def _random_walk(self, meta_molecule):
        first_node = list(meta_molecule.nodes)[0]
        meta_molecule.nodes[first_node]["position"] = np.array([0, 0, 0])
        vector_bundel = polyply.src.linalg_functions.norm_sphere(5000)

        for prev_node, current_node in nx.dfs_edges(meta_molecule, source=0):
            update_positions(vector_bundel, meta_molecule,
                             current_node, prev_node)

    def run_molecule(self, meta_molecule):
        self._random_walk(meta_molecule)
        return meta_molecule
