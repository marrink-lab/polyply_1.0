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
from .orient_by_bonds import orient_by_bonds
from .orient_by_frames import orient_by_frames
"""
Processor implementing a template based back
mapping to lower resolution coordinates for
a meta molecule.
"""

"""
BACKMAPPING_MODE contains a set of implemented backmapping modes.

Available modes:
    - automatically selecting a backmapping method (i.e. 'automatic') [UNDER CONSTRUCTION!!]
    - by optimizing bonded interactions (i.e. 'orient_by_bonds')
    - by user-defined frames (i.e. 'orient_by_frame')

"""

BACKMAPPING_MODE = {'automatic': orient_by_bonds, 'by-bonds': orient_by_bonds,
                    'by-frame': orient_by_frames}

class Backmap(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    places coordinates form a higher resolution createing positions
    for the lower resolution molecule associated with the MetaMolecule.
    """

    def __init__(self, fudge_coords=0.4, bmode='automatic', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fudge_coords = fudge_coords
        self.bmode = bmode

    def _place_init_coords(self, meta_molecule):
        """
        For each residue in a class:`polyply.src.MetaMolecule` the
        positions of the atoms associated with that residue stored in
        attr:`polyply.src.MetaMolecule.molecule` are created from a
        template residue located in attr:`polyply.src.MetaMolecule.templates`.

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.MetaMolecule`
        """
        built_nodes = []
        for node in meta_molecule.nodes:
            if  meta_molecule.nodes[node]["backmap"]:
                resname = meta_molecule.nodes[node]["resname"]
                cg_coord = meta_molecule.nodes[node]["position"]
                resid = meta_molecule.nodes[node]["resid"]
                high_res_atoms = meta_molecule.nodes[node]["graph"].nodes

                template = BACKMAPPING_MODE[self.bmode](meta_molecule, node,
                                                        meta_molecule.templates[resname],
                                                        built_nodes)

                for atom_high  in high_res_atoms:
                    atomname = meta_molecule.molecule.nodes[atom_high]["atomname"]
                    vector = template[atomname]
                    new_coords = cg_coord + vector * self.fudge_coords
                    meta_molecule.molecule.nodes[atom_high]["position"] = new_coords
                built_nodes.append(resid)

    def run_molecule(self, meta_molecule):
        """
        Apply placing of coordinates to meta_molecule. For more
        detail see `self._place_init_coords`.
        """
        self._place_init_coords(meta_molecule)
        return meta_molecule
