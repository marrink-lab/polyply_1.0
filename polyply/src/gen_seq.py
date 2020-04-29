import code
import readline
import rlcompleter
import networkx as nx
import json
from networkx.readwrite import json_graph
import polyply.src.console

"""
Console module provide `copen` method for opening interactive python shell in
the runtime.
"""

def copen(_globals, _locals):
    """
    Opens interactive console with current execution state.
    Call it with: `console.open(globals(), locals())`
    """
    context = _globals.copy()
    context.update(_locals)
    readline.set_completer(rlcompleter.Completer(context).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(context)
    shell.interact()

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


def gen_seq(args):
    print("\n\n")
    print("----------------------------------")
    print("Usage")
    print("Use the three commands below to")
    print("build a polymer sequence.\n")
    print("To do so first make a new block.")
    print("Then add residues to that block specificing,")
    print("how each new residue is connected to the one before.")
    print("Finally to generate n-monomers use the expand command,")
    print("followed by add to sequence and write_sequence.")
    print("\nCommands")
    print("new_block(name)")
    print("add_residue_to_block(name, resname, connections)")
    print("expand_block(name, n_monomers, connect)")
    print("add_block_to_squence(name, connection)")
    print("write_sequence(file_name)")
    print("\nTips")
    print("All variables name and resname need  to be ")
    print("entered using \" at the end and start of the name.\n")
    print("Connection are provided in the format:")
    print("[(resid1, resid2), (resid2, resid3) ]")
    print("----------------------------------")
    print("\n\n")

    global blocks
    global sequence
    blocks = Block("test")
    sequence = Sequence("seq")

    copen(globals(), locals())

    print('You created a residue graph!')
