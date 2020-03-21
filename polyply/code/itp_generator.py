import vermouth
import vermouth.molecule 
import vermouth.forcefield   
from vermouth.molecule import *
from vermouth.forcefield import *
from polyply.code.itp_read import *
from collections import Counter
import itertools
import collections
import numpy as np
#============================================================================================================================================================
#                                                                         Summary of Functions
#============================================================================================================================================================

def _treat_link_multiple(link_terms):

    count_terms = Counter(tuple(term.atoms) for term in link_terms)
    for term in link_terms:
        tag=count_terms[tuple(term.atoms)]
        if tag >= 1:
           term.meta.update({"version":tag})
           count_terms[tuple(term.atoms)] = tag -1

    return link_terms

def remove_dangling(block): 
   
    # Make sure to add the atomtype resdidue number etc to 
    # the proper nodes. 
   
    n_atoms = len(block.nodes) 
    link_interactions={} 
   
    for key in block.interactions: 
        link_terms=[] 
        block_terms=[] 
        for term in block.interactions[key]: 
            if np.sum(np.array(term.atoms) > n_atoms - 1) > 0: 
               link_terms.append(term) 
            else: 
               block_terms.append(term) 
        
        treated_link_terms=_treat_link_multiple(link_terms)      
        link_interactions.update({key:treated_link_terms}) 
        block.interactions[key] = block_terms 
   
    return link_interactions 


def add_link_attributes(graph, n_atoms, res_name, atom_name=None):
       '''
       Adds the minimum attributes to a vermout.molecule.Link
       so that DoLinks can do peform the matching. 
       '''
       last_resid=0
       atom_count=0
       res_count=0
       for idx in range(0,len(graph.nodes)):
           try:
               last_resid=graph.nodes[idx]["order"]
           except KeyError:
               nx.set_node_attributes(graph,{idx:res_count},"order")     
               nx.set_node_attributes(graph,{idx:res_name},"resname")
               atom_count += 1

               if atom_count == n_atoms:
                   res_count += 1
                   atom_count=0

def get_bonds(block): 
       bonds=[] 
       for interaction in block.interactions['bonds']: 
           bonds.append(interaction.atoms) 
       return np.array(bonds) 

def gen_itp(itp_files, link_file, modifications, n_monomers, out_name, name): 
    

    #1. read input parameters and populate force-field
    FF = ForceField('./')
    #block_names = []

    for file_handle in itp_files:
        with open(file_handle,'r') as _file:
             lines = _file.readlines()
             read_itp(lines,FF)        
        
    block_names = list(FF.blocks.keys())
          
    #2. get offsets 
    max_atoms=[]

    for name, n_trans in zip(block_names, n_monomers):
        n_atoms = len(FF.blocks[name].nodes)
        max_atoms.append(n_atoms * n_trans + sum(max_atoms))

    #3. construct graph from dangling bonds and/or links

      
    mol_graph_edges=[] 
    res_count=0
    print(n_trans,block_names)
    for name, n_trans in zip(block_names, n_monomers): 
        n_atoms = len([atom for atom in FF.blocks[name].atoms]) 
        bonds = get_bonds(FF.blocks[name]) 
        #v_sides = get_virtual_sides(FF.blocks[name])

        # something like this but we need to asses links somehow
        #try:
        #    link = np.array(FF.links[name].interactions['bonds'])
        #except KeyError:
        #    continue

        for i in np.arange(0,n_trans-1,1): 
            bonds_new = bonds + n_atoms * i             
            #mol_graph.add_edges_from(bonds_new) 
            mol_graph_edges.append(bonds_new)
        #add_mol_attributes(mol_graph, n_atoms, name)

            # add links
            # add bonds
 
        new_link_interactions = remove_dangling(FF.blocks[name])
        
        for key, interaction in new_link_interactions.items():
            new_link=vermouth.molecule.Link()
            new_link.name=name
            new_link.interactions.update({key:interaction})
            new_link.make_edges_from_interactions()
            add_link_attributes(new_link, n_atoms, name, atom_name=None)
            FF.links.append(new_link)
       
        res_count += 1 

    mol_graph_edges=np.array(mol_graph_edges).reshape(len(mol_graph_edges),2)
    #4. make molecule from blocks 
    
    new_mol = FF.blocks[block_names[0]].to_molecule()
    new_mol._force_field =FF
    new_mol.nrexcl = 1
    n_monomers[0] = n_monomers[0] - 1

    for name, n_trans in zip(block_names, n_monomers):
        for idx in range(0,n_trans):
            new_mol.merge_molecule(FF.blocks[name])

    new_mol.add_edges_from(mol_graph_edges)

    #5. add links
    sys = vermouth.System(force_field=FF) 
    sys.molecules.append(new_mol) 
    vermouth.DoLinks().run_system(sys)

    #6. apply 

    #vermouth.processors.do_post_trans(new_mol,FF)

    return sys.molecules[0]

# ToDo List
# 3.    test how endgroups can be implemented
# 5.    deal with virtual side graph construction
# 6.    deal with different number of exclusions    
# 7.    deal with doing the actual links between blocks
# 8.    make sure max_atoms is respected in generating the graphs 
# 9.    be smarter when counting the resid so that blocks that are more than one
# 10.FF.links.append(new_link)   residue are counted correctly
