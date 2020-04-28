import networkx as nx
import console

class BlockCollection:

      def __init__(self, name):
          self.name = name
          self.blocks = {}
          self.molecule = nx.Graph()
          self.current_block = None

def new_block(name):
    blocks.blocks[name] = nx.Graph()
    blocks.current_block = name

def add_residue(name, resname, connections):
    new_node = len(blocks.blocks[name].nodes) + 1
    blocks.blocks[name].add_node(new_node, resname=resname)
    blocks.blocks[name].add_edges_from(connections)

def remove_residue(name, resid):
    blocks[name].remove_node(resid)

def print_current_block(name):
    print("the following residues are part of this block:")
    for node in blocks.blocks[blocks.current_block].nodes:
        print("resid: ", node, ", resname:", blocks.blocks[blocks.current_block].nodes[node]["resname"])
    print("within the blocks the following residues are connected")
    for edge in blocks.blocks[blocks.current_block].edges:
        print("resids", edge[0], edge[1])

def expand(name, n_monomers, connect):
    mapping:
    for i in range(0, n_monomers):
        nx.relabel
    

print("\n\n\n\n")
print("Usage")
print("----------------------------------")
print("- new_block(name)")
print("- add_redidue(block_name, resname, connections)")
print("- expand(block_name, number of monomers)")
print("----------------------------------")
print("\n\n\n\n")

global blocks
blocks = BlockCollection("test")

console.copen(globals(), locals())

print('You created a residue graph!')


