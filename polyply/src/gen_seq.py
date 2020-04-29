import networkx as nx
import console
import json
from networkx.readwrite import json_graph

class Block:

    def __init__(self, name):
        self.name = name
        self.blocks = {}
        self.current_block = None

class Sequence:

    def __init__(self, name):
        self.name = name
        self.graph = nx.Graph()

def combine_graphs(graph1, graph2):
    offset = len(graph1.nodes)
    old_nodes = list(graph2.nodes)
    new_nodes = [node + offset for node in old_nodes]
    mapping = dict(zip(old_nodes, new_nodes))
    new_graph = nx.relabel_nodes(graph2, mapping)
    out_graph = nx.compose(new_graph, graph1)
    return out_graph

def new_block(name):
    blocks.blocks[name] = nx.Graph()
    blocks.current_block = name

def add_residue_to_block(name, resname, connections):
    new_node = len(blocks.blocks[name].nodes)
    blocks.blocks[name].add_node(new_node, resname=resname)
    blocks.blocks[name].add_edges_from(connections)

def add_block_to_squence(name, connection):
    block = blocks.blocks[name]
    sequence.graph = combine_graphs(sequence.graph, block)
    sequence.graph.add_edges_from(connection)

def remove_residue(name, resid):
    blocks.blocks[name].remove_node(resid)

def print_current_block():
    print("\nthe following residues are part of this block:")
    for node in blocks.blocks[blocks.current_block].nodes:
        print("resid: ", node, ", resname:", blocks.blocks[blocks.current_block].nodes[node]["resname"])
    print("\nwithin the block the following residues are connected:")
    for edge in blocks.blocks[blocks.current_block].edges:
        print("resids: ", edge[0]," , ", edge[1])

def expand_block(name, n_monomers, connect):
    block = blocks.blocks[name]
    new_block = block
    offset = 0
    for i in range(1, n_monomers):
        new_block = combine_graphs(new_block, block)
        for edge in connect:
            new_block.add_edge(edge[0]+offset, edge[1]+offset)
        offset = len(block.nodes)*i

    blocks.blocks[name] = new_block

def write_sequence(file_name):
    g = sequence.graph
    g_json = json_graph.node_link_data(g)
    json.dump(g_json, open(file_name,'w'), indent=1)

print("\n\n")
print("Usage")
print("----------------------------------")
print("Here goes good text")
print("----------------------------------")
print("\n\n")

global blocks
blocks = Block("test")
sequence = Sequence("seq")

console.copen(globals(), locals())

print('You created a residue graph!')


