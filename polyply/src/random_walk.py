import random
import networkx as nx
import numpy as np
import polyply
from .processor import Processor
from .linalg_functions import norm_sphere

def _take_step(vectors, step_length, coord):
    index = random.randint(0, len(vectors) - 1)
    new_coord = coord + vectors[index] * step_length
    return new_coord, index


def _is_overlap(meta_molecule, new_point, tol, fudge=1):

    for node in meta_molecule:
        try:
            coord = meta_molecule.nodes[node]["position"]
        except KeyError:
            continue

        if np.linalg.norm(coord - new_point) < tol * fudge:
           return True

    return False


def _combination(radius_A, radius_B):
    return (radius_A + radius_B) / 2.


def update_positions(vector_bundel, meta_molecule, current_node, prev_node):
    if "position" in meta_molecule.nodes[current_node]:
        return

    current_vectors = np.zeros(vector_bundel.shape)
    current_vectors[:] = vector_bundel[:]
    last_point = meta_molecule.nodes[prev_node]["position"]

    prev_resname = meta_molecule.nodes[prev_node]["resname"]
    current_resname = meta_molecule.nodes[current_node]["resname"]

    current_vdwr = meta_molecule.volumes[current_resname]
    prev_vdwr = meta_molecule.volumes[prev_resname]
    vdw_radius = _combination(current_vdwr, prev_vdwr)

    step_length = 2*vdw_radius

    while True:
        new_point, index = _take_step(vector_bundel, step_length, last_point)
        if not _is_overlap(meta_molecule, new_point, tol=vdw_radius):
            meta_molecule.nodes[current_node]["position"] = new_point
         #   print(meta_molecule.nodes[current_node]["resname"])
            break
        else:
            vector_bundel = np.delete(vector_bundel, index, axis=0)


class RandomWalk(Processor):
    """
    Add coordinates at the meta_molecule level
    through a random walk for all nodes which have
    build defined as true.
    """

    def _random_walk(self, meta_molecule):
        first_node = list(meta_molecule.nodes)[0]
        meta_molecule.nodes[first_node]["position"] = np.array([0, 0, 0])
        vector_bundel = norm_sphere(5000)
        for prev_node, current_node in nx.dfs_edges(meta_molecule, source=0):
            update_positions(vector_bundel, meta_molecule,
                             current_node, prev_node)

    def run_molecule(self, meta_molecule):
        self._random_walk(meta_molecule)
        return meta_molecule
