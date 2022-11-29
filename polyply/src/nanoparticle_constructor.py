import pprint
import numpy as np
from scipy.spatial import distance
import networkx as nx
import vermouth
import textwrap                                                                                                                                                                                                       
from pathlib import Path
import polyply.src.meta_molecule
from polyply.src.topology import Topology
from polyply.src.processor import Processor
from polyply.src.generate_templates import GenerateTemplates
from polyply import TEST_DATA
from polyply.src.meta_molecule import (MetaMolecule, Monomer)  
import polyply.src.map_to_molecule                                                                                                            
import polyply.src.polyply_parser    
from polyply.src.generate_templates import _expand_inital_coords                                                                                                                                                      
from polyply.src.minimizer import optimize_geometry                                                                                                                                                                  
from polyply.src.apply_links import MatchError, ApplyLinks      
from polyply.src.linalg_functions import center_of_geometry                                                                                                                                                          
from polyply.src.generate_templates import (find_atoms,                                                                                                                                  _expand_inital_coords,                                                                                                                       _relabel_interaction_atoms,                                                                                                                  compute_volume, map_from_CoG,                                                                                                                extract_block, GenerateTemplates,                                                                                                            find_interaction_involving)  

from vermouth.molecule import Interaction, Molecule, Block
from vermouth.gmx.itp import write_molecule_itp   
import vermouth.forcefield                                                                                                                   
import vermouth.molecule 
from vermouth.gmx.itp_read import read_itp                                                                        
from vermouth.molecule import Interaction

class NanoparticleGenerator(Processor):
    """
    initial setup of the class for nanoparticle compatible polyply generation 
    
    we take the core itp file (in this case scenario, we tried using a C60 core from the montielli 
    with the BCM ligand to create the PBCM nanoparticle with this approach.  
    """
    def __init__(self, core_itp_file, ligand_itp_file, ff_name, ligands_N, mode, ligand_anchor): 
        self.core_itp_file = core_itp_file # main NP itp file path 
        self.ligand_itp_file = ligand_itp_file # main ligand itp file path 
        self.ff_name = ff_name # force field name to store the blocks in 
        self.ligands_N = ligands_N # number of ligands to attach onto the nanoparticle core 
        self.force_field = vermouth.forcefield.ForceField(self.ff_name) # define forcefield name 
        self.top = Topology(self.force_field, name="test")
        self.mode = mode 
        self.ligand_anchor = ligand_anchor
        # initialize inherited class 
        #super().__init__(*args, **kwargs)

    def set_NP_ff(self, np_name):
        """
        register the nanoparticle core into the main force field representation
        then return the block of representing the nanoparticle resname 
        """
        with open(self.core_itp_file, 'r') as text_file:
            polyply.src.polyply_parser.read_polyply(text_file, self.force_field) 
            # create block representing the nanoparticle core 
            self.NP_block = self.force_field.blocks[np_name] # isolate the nanoparticle component
            
    def set_ligands_ff(self, residue_name):
        """
        read in the coordinates of the ligands to be added in as a block, which will be merged into the core 
        molecule object for the nanoparticle
        """
        #sample_sulfur_ligand_itp = "/home/sang/Desktop/polyply_1.0/test/C60/H_5C69BC.itp"
        if self.mode == None: 
            with open(self.ligand_itp_file, 'r') as text_file:
                polyply.src.polyply_parser.read_polyply(text_file, self.force_field) 
                # create block representing the ligand block 
                self.ligand_block = self.force_field.blocks[residue_name]
        else:
            #for itp_file in self.ligand_itp_file:
            with open(self.ligand_itp_file, 'r') as text_file:
                polyply.src.polyply_parser.read_polyply(text_file, self.force_field)
                self.ligand_block = self.force_field.blocks[residue_name]
                    
    def _identify_indices_for_core_attachement(self):
        """
        based on the coordinates of the NP core provided, 
        """
        # get initial coordinates of the block 
        init_coords = _expand_inital_coords(self.NP_block)
        # In the case of a spherical core, find the radius of the core 
        numpy_coords = np.asarray([init_coords[key] for key in init_coords.keys()])
        # find the center of geometry of the core 
        CoG = center_of_geometry(numpy_coords)
        # compute the radius 
        radius = distance.euclidean(CoG, numpy_coords[0])    
        # Find indices in the case of Janus and Striped particles
        length = radius * 2 
        minimum_threshold = min(numpy_coords[:,2])
        maximum_threshold = max(numpy_coords[:,2])
        # coordinates we wish to return 
        # the logic of this part takes from the original NPMDPackage code 
        if self.mode == None:
            core_values = {}
            for index, entry in enumerate(numpy_coords):
                core_values[index] = entry
            self.core_indices = [core_values]
            
        elif self.mode == "Striped":
            core_striped_values = {}
            core_ceiling_values = {}
            threshold = length / 3 # divide nanoparticle region into 3 
            striped_values = {}
            
            for index, entry in enumerate(numpy_coords):
                if entry[2] > minimum_threshold + threshold and entry[2] < maximum_threshold - threshold: 
                    core_striped_values[index] = entry  
                else: 
                    core_ceiling_values[index] = entry
            self.core_indices = [core_striped_values, core_ceiling_values]
            
        elif self.mode == "Janus":
            core_top_values = {}
            core_bot_values = {} 
            threshold = length / 2 # divide nanoparticle region into 2 
            for index, entry in enumerate(numpy_coords):
                if entry[2] > minimum_threshold + threshold:
                    core_top_values[index] = entry
                else:
                    core_bot_values[index] = entry 
            self.core_indices = [core_top_values, core_bot_values]
            
    def _identify_attachment_index(self):
        """
        Find index of atoms that corresponds to the atom on the ligand 
        which you wish to bond with the ligand on the nanoparticle core 
        """
        self.attachment_index = None
        for entry in self.ligand_block.atoms:
            if entry['atomname'] == self.ligand_anchor:
                self.attachment_index = int(entry['index'])
                break
                
    def _convert_to_vermouth_molecule(self):
        """
        we have two blocks representing the blocks of the NP core and 
        the ligand block. Hence, we wish to add links to the blocks  
        first add connection between the blocks 
        """
        self.np_molecule = self.NP_block.to_molecule()        
        ligand_len = len(list(self.ligand_block.atoms))
        # compute the max radius of the 
        core_size = max(list(self.NP_block))
        #self.attachment_index = 5 
        self._identify_indices_for_core_attachement() # generate indices we want to 
        self._identify_attachment_index()
        # tag on surface of the core to attach with ligands 
        core_index_relevant = self.core_indices[0] # first entry of the identified core indices 
        core_index_relevant_II = self.core_indices[1] # second entry of the identified core indices 
        interaction_list = [] 
        attachment_list = [] 
        # find the indices that are equivalent to the atom 
        # on the ligand we wish to attach onto the NP 
        # based on the ligands_N, and then merge onto the 
        # NP core the main ligand blocks ligand_N number of times 
        
        for index in range(0, self.ligands_N):
            ligand_index = index * ligand_len # compute starting index for each ligand
            scaled_ligand_index = ligand_index + core_size
            attachment_list.append(scaled_ligand_index)
            # merge the relevant molecule 
            self.np_molecule.merge_molecule(self.ligand_block)
        
        # create list of interactions for first batch of indices we have on the 
        # core surface 
        for index, index2 in enumerate(list(core_index_relevant.keys())[:self.ligands_N]):
            interaction = vermouth.molecule.Interaction(atoms=(index2, attachment_list[index] + self.attachment_index),
                                                parameters=['1', '0.0033', '50000'], 
                                                meta={})
            # append onto bonds the interactions between equivalent indices 
            self.np_molecule.interactions['bonds'].append(interaction)
        # create list for interactions for second batch of indices we have on the core surfaces
        #for index, index2 in enumerate(list(coreindex_relevant_II.keys())[:self.ligands_N]):
        #    interaction = vermouth.molecule.Interaction(atoms=(index2, atom_max + (index * length) + self.attachment_index),
        #                                        parameters=['1', '0.0033', '50000'], 
        #                                        meta={})
        #    # append onto bonds the interactions between equivalent indices 
        #    self.np_molecule.interactions['bonds'].append(interaction)
        for node in self.np_molecule.nodes:
            # change resname to moltype
            self.np_molecule.nodes[node]['resname'] = 'TEST' 
            
        self.graph = MetaMolecule._block_graph_to_res_graph(self.np_molecule) # generate residue graph 
        # generate meta molecule from the residue graph with a new molecule name 
        self.meta_mol = MetaMolecule(self.graph, force_field=self.force_field, mol_name = 'test')
        self.np_molecule.meta['moltype'] = 'TEST'
        # reassign the molecule with the np_molecule we have defined new interactions with 
        self.meta_mol.molecule = self.np_molecule
        #return self.meta_mol
        
    def create_itp(self, write_path):
        """
        create itp of the final concatenated ligand-functionalized nanoparticle file 
        """
        with open(str(write_path), 'w') as outfile:
            write_molecule_itp(self.np_molecule, outfile, header = header)

    def create_top(self):
        """
        the idea here is to add onto the topology object the molecules within the nanoparticle object 
        and generate a topology that can be converted to a .gro file - unfortunately, not working at the 
        moment 
        """
        self.top.molecules = [self.meta_mol]
        pos_dict = nx.spring_layout(self.top.molecules[0].molecule, dim=3)
        nx.set_node_attributes(self.top.molecules[0].molecule, pos_dict, 'position')

    def create_gro(self, write_path):
        """
        create gro file from the system 
        """
        self.top.box = np.array([3.0, 3.0, 3.0])
        system = self.top.convert_to_vermouth_system()
        vermouth.gmx.gro.write_gro(system, write_path, precision=7,
                           title='these are random positions', box=self.top.box, defer_writing=False)
    def optimize_geometry(self):
        """
        this function should take the newly attached carbon nanoparticle with the sulfur 
        ligand (not realistic, but wanting to check the functionality)
        """
        for _iteration in range(0, 10):
            init_coords = _expand_inital_coords(self.NP_block)
            success, coords = optimize_geometry(self.NP_block, init_coords, ["bonds", "constraints", "angles"])
            success, coords = optimize_geometry(self.NP_block, coords, ["bonds", "constraints", "angles", "dihedrals"])
            if success:
                print("success!")
                break      
                           
if __name__ == '__main__':

    # building fullerene PBCM complex - need necessary itp paths 
    PATH = "/home/sang/Desktop/git/polyply_1.0/polyply/data/nanoparticle_test/PCBM"
    NP_itp = PATH + "/" + "C60.itp" 
    BCM_itp = PATH + "/" + "BCM.itp"
    # In this case, as the C60 core is relatively small compared to the ligand size, the
    # system will not work 
    BCM_class = NanoparticleGenerator(NP_itp, BCM_itp, 'test', 1, 'Janus', 'C07')
    BCM_class.set_NP_ff('C60')
    BCM_class.set_ligands_ff('BCM')
    BCM_class._convert_to_vermouth_molecule()
    BCM_class.create_top()
    BCM_class.create_itp(PATH + "/" + "NP.itp")
    BCM_class.create_gro(PATH + "/" + "NP.gro")
    

