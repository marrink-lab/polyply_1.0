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
import metapredict as meta
from .simple_seq_parsers import parse_txt, parse_ig, parse_fasta, parse_json
import numpy as np

ONE_LETTER_AA = {"G": "GLY",
                 "A": "ALA",
                 "V": "VAL",
                 "C": "CYS",
                 "P": "PRO",
                 "L": "LEU",
                 "I": "ILE",
                 "M": "MET",
                 "W": "TRP",
                 "F": "PHE",
                 "S": "SER",
                 "T": "THR",
                 "Y": "TYR",
                 "N": "ASN",
                 "Q": "GLN",
                 "K": "LYS",
                 "R": "ARG",
                 "H": "HIS",
                 "D": "ASP",
                 "E": "GLU",
                 }

THREE_LETTER_AA = {ONE_LETTER_AA[key]: key for key in ONE_LETTER_AA.keys()}

class MakeIDP(Processor):
    def check_seq(self, sequence=None, sequence_file=None, idp_override=False):

        if sequence:
            single_letter_seq = ''
            for monomer in sequence:
                single_letter_seq += THREE_LETTER_AA[monomer[0]]*monomer[1]


        #the following bodged from elsewhere
        elif sequence_file:
            parsers = {"txt": parse_txt,
                       "fasta": parse_fasta,
                       "ig": parse_ig,
                       "json": parse_json, }

            single_letter_seq = ''
            #This is probably not the most efficient way to do it but it shouldn't reduce program speed substantially
            extension = sequence_file.suffix.casefold()[1:]
            if extension in parsers:
                graph = parsers[extension](sequence_file)
                for node in graph.nodes:
                    single_letter_seq += THREE_LETTER_AA[graph.nodes[node]["resname"]]
            else:
                msg = f"File format {extension} is unknown."
                raise IOError(msg)

        if len(single_letter_seq) >0:
            disorder = meta.predict_disorder_domains(single_letter_seq)
            seq_len = len(single_letter_seq)

            #Firstly, terminate the program if there are any ordered domains in the sequence
            if len(disorder.folded_domain_boundaries)>0:
                msg = "Sequence doesn't appear disordered, will terminate here"
                raise IOError(msg)

            # if there are no folded domains in the sequence, that should mean that there is only a single disordered one
            # check that this is indeed the case, and then end this check
            assert disorder.disordered_domain_boundaries[0][1] == seq_len
            return("Sequence looks disordered to me, will proceed with modifications")


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