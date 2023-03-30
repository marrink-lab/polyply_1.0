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

class MakeIDP(Processor):

    def run_molecule(self, meta_molecule):
        #establish which nodes of the molecule are the BB beads.
        BB_inds = [j for j in meta_molecule.molecule.nodes if meta_molecule.molecule.nodes[j]['atomname'] == 'BB']

        #add the backbone dihedrals
        for i in range(len(BB_inds) - 3):
            inds = BB_inds[i:i + 4]

            meta_molecule.molecule.add_interaction('dihedrals',
                                                   atoms=inds,
                                                   parameters=['9', '60', '1.35', '1'],
                                                   meta = {'group': 'IDP BB dih'}
                                                   )

            meta_molecule.molecule.add_interaction('dihedrals',
                                                   atoms=inds,
                                                   parameters=['9', '95', '0.30', '1'],
                                                   meta={'group': 'IDP BB dih'}
                                                   )

            meta_molecule.molecule.add_interaction('dihedrals',
                                                   atoms=inds,
                                                   parameters=['9', '130', '-1.50', '2'],
                                                   meta={'group': 'IDP BB dih'}
                                                   )

        #add the virtual sites in
        #for some reason this breaks if we replace all mentions of current_len in the loop with this line?
        current_len = len(meta_molecule.molecule)
        for i in BB_inds:

            meta_molecule.molecule.add_node(current_len,
                                            index = current_len,
                                            atomname = 'CA',
                                            atype = 'VS',
                                            resname = meta_molecule.molecule.nodes[i]['resname'],
                                            resid = meta_molecule.molecule.nodes[i]['resid'],
                                            charge_group = current_len+1,
                                            charge = 0.0,
                                            mass = 0.0
                                            )

            meta_molecule.molecule.add_interaction('virtual_sitesn',
                                                   atoms=[current_len,
                                                          i],
                                                   parameters=['1'],
                                                   meta = {'group': 'VS for increased IDP interactions'}
                                                   )

            current_len += 1

        return meta_molecule