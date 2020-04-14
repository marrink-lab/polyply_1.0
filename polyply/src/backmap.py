import numpy as np
from .processor import Processor


class Backmap(Processor):


    def _place_init_coords(self, meta_molecule):
        coords = np.zeros((len(meta_molecule.molecule.nodes), 3))
        idx = 0
        for node in meta_molecule.nodes:
            resname = meta_molecule.nodes[node]["resname"]
            CoG = meta_molecule.nodes[node]["position"]
            template = meta_molecule.templates[resname]
            for atom, vector in template.items():
                new_coords = CoG + vector
                coords[idx:(idx+1), :] = new_coords
                idx += 1
        meta_molecule.coords = coords

    def run_molecule(self, meta_molecule):
        self._place_init_coords(meta_molecule)
        return meta_molecule
