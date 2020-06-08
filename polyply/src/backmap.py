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

from .processor import Processor
from .generate_templates import find_atoms
"""
Processor implementing a template based back
mapping to lower resolution coordinates for
a meta molecule.
"""

class Backmap(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    places coordinates form a higher resolution createing positions
    for the lower resolution molecule associated with the MetaMolecule.
    """
    @staticmethod
    def _place_init_coords(meta_molecule):
        """
        For each residue in a class:`polyply.src.MetaMolecule` the
        positions of the atoms associated with that residue stored in
        attr:`polyply.src.MetaMolecule.molecule` are created from a
        template residue located in attr:`polyply.src.MetaMolecule.templates`.

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.MetaMolecule`
        """
        count = 0
        for node in meta_molecule.nodes:
            resname = meta_molecule.nodes[node]["resname"]
            cg_coord = meta_molecule.nodes[node]["position"]
            template = meta_molecule.templates[resname]
            resid = node + 1
            low_res_atoms = find_atoms(meta_molecule.molecule, "resid", resid)
            for atom_low  in low_res_atoms:
                atomname = meta_molecule.molecule.nodes[atom_low]["atomname"]
                vector = template[atomname]
                new_coords = cg_coord + vector
                if meta_molecule.molecule.nodes[atom_low]["build"]:
                    meta_molecule.molecule.nodes[atom_low]["position"] = new_coords

    def run_molecule(self, meta_molecule):
        """
        Apply placing of coordinates to meta_molecule. For more
        detail see `self._place_init_coords`.
        """
        self._place_init_coords(meta_molecule)
        return meta_molecule
