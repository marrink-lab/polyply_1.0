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

from networkx.algorithms import isomorphism 
import networkx as nx
#============================================================================================================================================================
#                                                                         Summary of Functions
#============================================================================================================================================================

class meta_molecule(nx.Graph):
      """
      Graph that describes molecules at the residue level.
      """

      def __init__(self,**args, **kwargs):
          self._force_field = kwargs.pop('force_field',None)
          self.name         = kwargs.pop('name',None)
          self.molecule     = vermouth.Molecuel(nx.Graph())

      @classmethod
      def from_polyply(cls):




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
       for node in graph.nodes:

           try:
               last_resid=graph.nodes[node]["order"]
           except KeyError:
               res_count = int(node/n_atoms)

               nx.set_node_attributes(graph,{node:res_count},"order")
               nx.set_node_attributes(graph,{node:res_name},"resname")

       return graph

def get_interactions(block, names):
       interactions=[]
       for name in names:
           for interaction in block.interactions[name]:
               interactions.append(interaction.atoms)

       return np.array(interactions)

def neighborhood(G, node, n):
#     Adobted from: https://stackovrflow.com/questions/22742754/finding-the-n-degree-neighborhood-of-a-node    
    path_lengths = nx.single_source.dijkstra_path_length(G, node)
    neighbours=[ node for node, length in path_lengths.items() if length <= n and length > 1]
    return(neighbours)

def construct_bonded_exclusions(molecule, nrexcl):
    exclusions=[]

    for atom in molecule.nodes:
        excl_atoms = neighborhood(molecule, atom, nrexcl)
        for atom_B in excl_atoms:
            exclusions.append((atom,atom_B))

    return exclusions

def do_exclusions(molecule,nr_exclusions):
    exclusions=[]
    for key, nrexcl in nr_exclusions.items():
        raw_exclusions = construct_bonded_exclusions(molecule,nrexcl)
        allowed_nodes = list(nx.get_node_attributes(molecule, "resname").keys())
        for exclusion in raw_exclusions:
            if exclusion[0] in allowed_nodes and exclusion[1] in allowed_nodes:

               interaction = Interaction(
                  atoms=exclusion,
                  parameters=[],
                  meta={},
                  )

               exclusions.append(interaction)
    old_excl = molecule.interactions["exclusions"]
    new_excl = exclusions + old_excl
    molecule.interactions["exclusions"] = new_excl

def gen_itp(itp_files, link_file, modifications, n_monomers, out_name, name):


    #1. read input parameters and populate force-field
    FF = ForceField('./')
    #block_names = []

    for file_handle in itp_files:
        with open(file_handle,'r') as _file:
             lines = _file.readlines()
             read_itp(lines,FF)

    block_names = list(FF.blocks.keys())

    with open(link_file,'r') as _file:
         lines = _file.readlines()
         read_ff(lines,FF)

    #2. get offsets
    max_atoms=[]

    for name, n_trans in zip(block_names, n_monomers):
        n_atoms = len(FF.blocks[name].nodes)
        max_atoms.append(n_atoms * n_trans + sum(max_atoms))

    #3. construct graph from dangling bonds and/or links

    #print(max_atoms)
    mol_graph_edges=[]
    res_count=0
#    print(n_trans,block_names)
    offset=0
    for name, n_trans, limit in zip(block_names, n_monomers, max_atoms):
        n_atoms = len([atom for atom in FF.blocks[name].atoms])
        pseudo_bonds = get_interactions(FF.blocks[name],["constraints","bonds"])
        print(pseudo_bonds)
        for i in np.arange(0,n_trans,1):
            offset += n_atoms * i
            pseudo_bonds_new = pseudo_bonds + offset
            pseudo_bonds_new = pseudo_bonds_new[np.all(pseudo_bonds_new<limit,(1)) ]
            mol_graph_edges.append(pseudo_bonds_new)
            res_count += 1

        #print(mol_graph_edges)

    #4. Split Polyply into blocks and links

        new_link_interactions = remove_dangling(FF.blocks[name])

        res_name = FF.blocks[name].nodes[0]['resname']
        for key, interactions in new_link_interactions.items():
           for interaction in interactions:
               new_link=vermouth.molecule.Link()
               new_link.name=name
               new_link.interactions.update({key:[interaction]})
               new_link.make_edges_from_interaction_type(type_=key)
               new_link = add_link_attributes(new_link, n_atoms, res_name, atom_name=None)
               FF.links.append(new_link)

    mol_graph_edges=np.vstack(tuple(mol_graph_edges))
    #print(mol_graph_edges)

    #5. make molecule from blocks
    new_mol = FF.blocks[block_names[0]].to_molecule()
    new_mol._force_field =FF
    new_mol.nrexcl = 1
    n_monomers[0] = n_monomers[0] - 1

    exclusions={}
    for name, n_trans in zip(block_names, n_monomers):
        exclusions[name] =  FF.blocks[name].nrexcl
        FF.blocks[name].nrexcl = 1
        for idx in range(0,n_trans):
            new_mol.merge_molecule(FF.blocks[name])


    # add links between dangling nodes
    sub_graphs = list(nx.connected_components(new_mol))

    for idx in range(0,len(sub_graphs)-1):
        # this presupposes that the last atom and the first atom of a block
        # have the same resname as the link that is to be applied
        link_name = max(sub_graphs[idx]) + "_" + min(sub_graphs[idx+1])
        links = get_links_by_name(FF.links,link_name)
        for link in links:
            for presudo_bond in link.interactions['bonds']:
               

    new_mol.add_edges_from(mol_graph_edges)
    do_exclusions(new_mol,exclusions)




    #6. add links
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
# 10a.FF.links.append(new_link)   residue are counted correctly
# build in guard against diverging resnames and block-names
# guard against interactions that are not along bonds i.e. 1-2-3-4 and an angle 1 2 4. 
#
