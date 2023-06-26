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

def GLY_sorting(GLY_resids, BB_resids, BB_inds, SC1_resids, SC1_inds):
    '''
    this is a horrendous implementation
    
    this function makes lists of SBBB and BBBS dihedrals, the scfix needed when we have GLYs
    
    However, in doing this we need to check:
        1) How many we have in the protein all together
        2) If we have consecutive GLYs
    
    The final 'else' loop on its own is relatively clear about how this is done.
    
    There's definitely a better way to do all this, but THIS IS ROBUST AND WORKS (and without noticable time increases) for now.
    
    '''

    scfix_SC1_BB_BB_BB  = []
    scfix_BB_BB_BB_SC1  = []
    
    if len(GLY_resids) == 0:
        return scfix_BB_BB_BB_SC1, scfix_SC1_BB_BB_BB
    
    elif len(GLY_resids) == 1:    
        for k,l in enumerate(GLY_resids):
            if (l > 0) and (l<BB_resids[-1]):
                BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                
                BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
                SC_m1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l-1)[0][0]]
        
                BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                SC_p1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l+1)[0][0]]
        
                to_append_SBBB = [SC_m1, BB_m1, BB_0, BB_p1]
                to_append_BBBS = [BB_m1, BB_0, BB_p1, SC_p1]

            if len(to_append_SBBB) == 4:
                scfix_SC1_BB_BB_BB.append(to_append_SBBB)
        
            if len(to_append_BBBS) == 4:
                scfix_BB_BB_BB_SC1.append(to_append_BBBS)
            
        return scfix_BB_BB_BB_SC1, scfix_SC1_BB_BB_BB

    elif len(GLY_resids) == 2:
        # check that they're not next to each other
        if GLY_resids[1] - GLY_resids[0] != 1:        
            for k,l in enumerate(GLY_resids):
                if (l > 0) and (l<BB_resids[-1]):
                    BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                    
                    BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
                    SC_m1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l-1)[0][0]]
            
                    BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                    SC_p1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l+1)[0][0]]
        
                    to_append_SBBB = [SC_m1, BB_m1, BB_0, BB_p1]
                    to_append_BBBS = [BB_m1, BB_0, BB_p1, SC_p1]                    
    
                if len(to_append_SBBB) == 4:
                    scfix_SC1_BB_BB_BB.append(to_append_SBBB)
            
                if len(to_append_BBBS) == 4:
                    scfix_BB_BB_BB_SC1.append(to_append_BBBS)
            
            return scfix_BB_BB_BB_SC1, scfix_SC1_BB_BB_BB
        
        else: # ie. if they are in fact next to each other
            for k,l in enumerate(GLY_resids):
                if (l > 0) and (l<BB_resids[-1]):              
                    #check if the next one is also GLY, then we can only apply SBBB.                
                    #also ensure we're not at the last GLY 
                    if (l<GLY_resids[-1]):
                    
                        BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                        
                        BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
                        SC_m1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l-1)[0][0]]
                
                        BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                      
                        to_append_SBBB = [SC_m1, BB_m1, BB_0, BB_p1]
        
                    #check behind, if it is then we can only apply BBBS
                    #also ensure we're not at the first GLY
                    elif (l>GLY_resids[0]):    
                
                        BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                        
                        BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
                      
                        BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                        SC_p1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l+1)[0][0]]
                        
                        to_append_BBBS = [BB_m1, BB_0, BB_p1, SC_p1]
                  
                if len(to_append_SBBB) == 4:
                    scfix_SC1_BB_BB_BB.append(to_append_SBBB)
            
                if len(to_append_BBBS) == 4:
                    scfix_BB_BB_BB_SC1.append(to_append_BBBS)

            return scfix_BB_BB_BB_SC1, scfix_SC1_BB_BB_BB
    
    else:
        for k,l in enumerate(GLY_resids):
            to_append_SBBB = []
            to_append_BBBS = []
            
            
            #shouldn't apply this if the GLY is either terminal residue in the protein
            #NB resids start from 1
            if (l > 1) and (l<BB_resids[-1]):

                if l < GLY_resids[-1]:
                    #if we're in a string of 3 consecutive GLY then break because we can't apply anything
                    #also ensure that we're not at either the first or last GLY in the sequence, otherwise we can't find the previous or next GLY
                    if (GLY_resids[k+1] - GLY_resids[k] == 1) and (GLY_resids[k] - GLY_resids[k-1] == 1) and (l>GLY_resids[0]) and (l<GLY_resids[-1]):
                        to_append_SBBB = []
                        to_append_BBBS = []
                        
                    #check if the next one is also GLY, then we can only apply SBBB.                
                    #also ensure we're not at the last GLY 
                    elif (GLY_resids[k+1] - GLY_resids[k] == 1) and (l<GLY_resids[-1]):
                    
                        BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                        
                        BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
                        SC_m1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l-1)[0][0]]
                
                        BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                      
                        to_append_SBBB = [SC_m1, BB_m1, BB_0, BB_p1]
            
                    #check behind, if it is then we can only apply BBBS
                    #also ensure we're not at the first GLY
                    elif (GLY_resids[k] - GLY_resids[k-1] == 1) and (l>GLY_resids[0]):    
                
                        BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                        
                        BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
                      
                        BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                        SC_p1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l+1)[0][0]]
                        
                        to_append_BBBS = [BB_m1, BB_0, BB_p1, SC_p1]
        
                    #if this is a single GLY then we can apply both SBBB and BBBS
                    else:            
                        BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                        
                        BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
                        SC_m1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l-1)[0][0]]
                
                        BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                        SC_p1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l+1)[0][0]]
            
                        to_append_SBBB = [SC_m1, BB_m1, BB_0, BB_p1]
                        to_append_BBBS = [BB_m1, BB_0, BB_p1, SC_p1]
                
                #once we get to the final one, check behind or apply normally.
                else:
                    
                    #check behind, if it is then we can only apply BBBS
                    #also ensure we're not at the first GLY
                    if (GLY_resids[k] - GLY_resids[k-1] == 1) and (l>GLY_resids[0]):    
                
                        BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                        
                        BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
                      
                        BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                        SC_p1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l+1)[0][0]]
                        
                        to_append_BBBS = [BB_m1, BB_0, BB_p1, SC_p1]
        
                    #if this is a single GLY then we can apply both SBBB and BBBS
                    else:            
                        BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                        
                        BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
                        SC_m1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l-1)[0][0]]
                
                        BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                        SC_p1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l+1)[0][0]]
            
                        to_append_SBBB = [SC_m1, BB_m1, BB_0, BB_p1]
                        to_append_BBBS = [BB_m1, BB_0, BB_p1, SC_p1]
                
            else:
                to_append_SBBB = []
                to_append_BBBS = []    
                
            if len(to_append_SBBB) == 4:
                scfix_SC1_BB_BB_BB.append(to_append_SBBB)
        
            if len(to_append_BBBS) == 4:
                scfix_BB_BB_BB_SC1.append(to_append_BBBS)
        
        return scfix_BB_BB_BB_SC1, scfix_SC1_BB_BB_BB


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

            #Check if there are any folded domains in the sequence
            if len(disorder.folded_domain_boundaries)>0:
                #if there are but the override flag is given, ignore this
                if idp_override == True:
                    return("Sequence doesn't appear disordered but will ignore")
                #otherwise terminate the parameter generation here with a warning
                else:
                    msg = "Sequence doesn't appear disordered, will terminate here"
                    raise IOError(msg)

            # if there are no folded domains in the sequence, that should mean that there is only a single disordered one
            # check that this is indeed the case, and then end this check
            else:
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
                                                    parameters=['9', '-120', '-1.00', '1'],
                                                    meta = {'group': 'IDP BB dih'}
                                                    )

            meta_molecule.molecule.add_interaction('dihedrals',
                                                    atoms=inds,
                                                    parameters=['9', '120', '-1.00', '2'],
                                                    meta={'group': 'IDP BB dih'}
                                                    )

        #add the virtual sites in
        #for some reason this breaks if we replace all mentions of current_len in the loop with this line?
        current_len = len(meta_molecule.molecule)
        
        # need to make a list of vs site indices to create an exclusion list
        VS_list = np.zeros((0,2))
        
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
                                                   
            meta_molecule.molecule.add_interaction('position_restraints',
                                                   atoms=[i],
                                                   parameters=['1', '1000', '1000', '1000'],
                                                   meta = {'ifdef': 'POSRES'}
                                                   )

            VS_list = np.vstack((VS_list, np.array([current_len, i])))
            
            current_len += 1
        
        #exclude interactions between VS and BB beads built on adjacent beads
        for i,j in enumerate(VS_list):
            if i == 0:
                meta_molecule.molecule.add_interaction('exclusions',
                                                        atoms=[VS_list[i][0], VS_list[i][1], VS_list[i+1][1]],
                                                        parameters = [],
                                                        meta = {'group': 'IDP VS exclusions'})
            elif i == len(VS_list)-1:
                meta_molecule.molecule.add_interaction('exclusions',
                                                        atoms=[VS_list[i][0], VS_list[i-1][1], VS_list[i][1]],
                                                        parameters = [],
                                                        meta = {'group': 'IDP VS exclusions'})
            else:
                meta_molecule.molecule.add_interaction('exclusions',
                                                        atoms=[VS_list[i][0], VS_list[i-1][1], VS_list[i][1], VS_list[i+1][1]],
                                                        parameters = [],
                                                        meta = {'group': 'IDP VS exclusions'})

                
        #sort out the IDP scfix
        #first get the resids and the indices of the SC1 beads in the protein
        SC1_inds = [j for j in meta_molecule.molecule.nodes if meta_molecule.molecule.nodes[j]['atomname'] == 'SC1']
        SC1_resids = [meta_molecule.molecule.nodes[j]['resid'] for j in meta_molecule.molecule.nodes if meta_molecule.molecule.nodes[j]['atomname'] == 'SC1']
        BB_resids = [meta_molecule.molecule.nodes[j]['resid'] for j in meta_molecule.molecule.nodes if meta_molecule.molecule.nodes[j]['atomname'] == 'BB']
        
        #find the GLY (ie, residues with no side chain) to apply the SC-BB-BB-BB and BB-BB-BB-SC scfix to
        GLY_resids = [meta_molecule.molecule.nodes[j]['resid'] for j in meta_molecule.molecule.nodes if meta_molecule.molecule.nodes[j]['resname'] == 'GLY']
        #because of the addition of virtual sites, each residue is now counted twice. So just take the first half of the list.
        GLY_resids = GLY_resids[:int(len(GLY_resids)/2)]
        
        scfix_SC1_BB_BB_SC1 = []
        for k,l in enumerate(SC1_resids[:-1]):
            #make sure the residues are consecutive
            if SC1_resids[k+1] - SC1_resids[k] == 1:
                
                #get the atom numbers of the 4 atoms we need
                SC1_1 = SC1_inds[k]
                BB_1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                BB_2 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                SC1_2 = SC1_inds[k+1]

                to_append_SBBS = [SC1_1, BB_1, BB_2, SC1_2]
                if len(to_append_SBBS) == 4:
                    scfix_SC1_BB_BB_SC1.append(to_append_SBBS)
                    
        scfix_BB_BB_BB_SC1, scfix_SC1_BB_BB_BB = GLY_sorting(GLY_resids, BB_resids, BB_inds, SC1_resids, SC1_inds)


        '''
        leaving this commented for now because it's possibly easier to understand than the mess of a function above,
        even though I *think* the function now handles all possible edge cases
        '''
        # scfix_SC1_BB_BB_BB  = []
        # scfix_BB_BB_BB_SC1  = []
        
        # for k,l in enumerate(GLY_resids):
        #     '''
        #     this is a horrendous implementation
        #     '''
            
        #     to_append_SBBB = []
        #     to_append_BBBS = []
            
        #     #shouldn't apply this if the GLY is either terminal residue in the protein
        #     if (l > 0) and (l<BB_resids[-1]):

        #         #if we're in a string of 3 consecutive GLY then break because we can't apply anything
        #         #also ensure that we're not at either the first or last GLY in the sequence, otherwise we can't find the previous or next GLY
        #         if (GLY_resids[k+1] - GLY_resids[k] == 1) and (GLY_resids[k] - GLY_resids[k-1] == 1) and (l<GLY_resids[-1]) and (l>GLY_resids[0]):
        #             break
                
        #         #check if the next one is also GLY, then we can only apply SBBB.                
        #         #also ensure we're not at the last GLY 
        #         elif (GLY_resids[k+1] - GLY_resids[k] == 1) and (l<GLY_resids[-1]):
                
        #             BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                    
        #             BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
        #             SC_m1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l-1)[0][0]]
            
        #             BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                  
        #             to_append_SBBB = [SC_m1, BB_m1, BB_0, BB_p1]

        #         #check behind, if it is then we can only apply BBBS
        #         #also ensure we're not at the first GLY
        #         elif (GLY_resids[k] - GLY_resids[k-1] == 1) and (l>GLY_resids[0]):    
            
        #             BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                    
        #             BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
                  
        #             BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
        #             SC_p1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l+1)[0][0]]
                    
        #             to_append_BBBS = [BB_m1, BB_0, BB_p1, SC_p1]

        #         #if this is a single GLY then we can apply both SBBB and BBBS
        #         else:            
        #             BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                    
        #             BB_m1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l-1)[0][0]]
        #             SC_m1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l-1)[0][0]]
            
        #             BB_p1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
        #             SC_p1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l+1)[0][0]]
        
        #             to_append_SBBB = [SC_m1, BB_m1, BB_0, BB_p1]
        #             to_append_BBBS = [BB_m1, BB_0, BB_p1, SC_p1]

        #     if len(to_append_SBBB) == 4:
        #         scfix_SC1_BB_BB_BB.append(to_append_SBBB)
        
        #     if len(to_append_BBBS) == 4:
        #         scfix_BB_BB_BB_SC1.append(to_append_BBBS)

        
        scfix_BB_BB_SC1 = []
        scfix_SC1_BB_BB = []
        for k,l in enumerate(BB_resids[:-1]):
            if l+1 not in GLY_resids:
                BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                BB_1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                SC_1 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l+1)[0][0]]
    
                to_append_BBS = [BB_0, BB_1, SC_1]
                if len(to_append_BBS) == 3:
                    scfix_BB_BB_SC1.append(to_append_BBS)
            
            if l not in GLY_resids:
                SC_0 = SC1_inds[np.where(np.array(SC1_resids, dtype = int) == l)[0][0]]
                BB_0 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l)[0][0]]
                BB_1 = BB_inds[np.where(np.array(BB_resids, dtype = int) == l+1)[0][0]]
                                    
                to_append_SBB = [SC_0, BB_0, BB_1]
                if len(to_append_SBB) == 3:
                    scfix_SC1_BB_BB.append(to_append_SBB)
        
        #remove the default martini BBS interactions to begin
        original_bbs_angles = [i for i in meta_molecule.molecule.interactions['angles'] if i.meta['group'] == 'BBS angles regular martini']
        for i in original_bbs_angles:
            meta_molecule.molecule.remove_interaction('angles',
                                                      i.atoms,
                                                      version = 1)
        
        #now add the IDP scfix interactions in for BBS and SBB angles
        for i in scfix_BB_BB_SC1:
            meta_molecule.molecule.add_interaction('angles',
                                                    atoms=i,
                                                    parameters=['10', '85', '10'],
                                                    meta = {'group': 'IDP BB-BB-SC1 scfix'}
                                                    )

        for i in scfix_SC1_BB_BB:
            meta_molecule.molecule.add_interaction('angles',
                                                    atoms=i,
                                                    parameters=['10', '85', '10'],
                                                    meta = {'group': 'IDP SC1-BB-BB scfix'}
                                                    )
        
        #add the SC1-BB-BB-SC1 dihedrals where we need them
        for i in scfix_SC1_BB_BB_SC1:
            
            meta_molecule.molecule.add_interaction('dihedrals',
                                                    atoms=i,
                                                    parameters=['9', '-130', '-1.50', '1'],
                                                    meta = {'group': 'IDP SC1-BB-BB-SC1 dihedrals'}
                                                    )

            meta_molecule.molecule.add_interaction('dihedrals',
                                                    atoms=i,
                                                    parameters=['9', '100', '-1.50', '2'],
                                                    meta={'group': 'IDP SC1-BB-BB-SC1 dihedrals'}
                                                    )
        
        #add the SC1-BB-BB-BB dihedrals where we need them
        for i in scfix_SC1_BB_BB_BB:
            meta_molecule.molecule.add_interaction('dihedrals',
                                                    atoms = i,
                                                    parameters = ['1', '115', '-4.50', '1'],
                                                    meta={'group': 'IDP SC1-BB-BB-BB dihedrals'}
                                                    )
    
        # add the BB-BB-BB-SC1 dihedrals where we need them
        for i in scfix_BB_BB_BB_SC1:
           meta_molecule.molecule.add_interaction('dihedrals',
                                                   atoms = i,
                                                   parameters = ['1', '0', '-2.00', '1'],
                                                   meta={'group': 'IDP BB-BB-BB-SC1 dihedrals'}
                                                   )            
        
        
        return meta_molecule
