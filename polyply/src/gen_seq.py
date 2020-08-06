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

import random
import json
import networkx as nx
from networkx.readwrite import json_graph
from .load_library import load_library
from .generate_templates import find_atoms


def _branched_graph(resname, branching_f, n_levels):
    """
    Generate a tree graph of branching factor `branching_f`
    and number of generations as `n_levels` - 1. Each node
    gets the attribute resname.

    Parameters
    ----------
    resname:  str
    branching_f: int
    n_levels: int

    Returns
    -------
    `:class:networkx.Graph`
    """
    graph = nx.balanced_tree(r=branching_f, h=n_levels-1)
    resnames = {idx: resname for idx in graph.nodes}
    nx.set_node_attributes(graph, resnames, "resname")
    return graph

def _random_replace_nodes(graph, residues, weights):
    """
    Randomly replace resname attribute of `graph`
    based on the names in `residues` taking into
    account `weights` provided.

    Parameters
    ----------
    graph: `:class:networkx.Graph`
    residues: list[str]
    weights: list[float]

    Returns:
    --------
    `:class:networkx.Graph`
    """
    for node in graph.nodes:
        resname = random.choices(residues, weights=weights)
        graph.nodes[node]["resname"] = resname[0]

    return graph

# def _expand_res_graph(graph, macros, connects):
#   """
#   Expand a `graph` if the node name is in `macros`
#   and add connect records as defined in connects.

#   Parameters
#   ----------
#   graph: `:class:networkx.Graph`
#   macros: dict[`class.MacroString`]
#   connects: dict[(resname, resname)]

#   Returns:
#   -------
#   `:class:networkx.Graph`
#   """
#   expanded_graph = nx.Graph()
#   had_edge = ()

#   #start

#   for prev_node, current_node in nx.dfs_edges(graph, source=0):
#       current_tag = graph.nodes[current_node]["resname"]
#       if current_tag in macros:
#          prev_tag = graph.nodes[edge[1]]["resname"]
#          macro = macros[tag].gen_graph(macros, connects)
#          expanded_graph = nx.disjoint_union(expanded_graph, macro)

#          if (current_tag, prev_tag) in connects:
#            current_resid, prev_resid = connects[(current_tag, prev_tag)]
#          elif (prev_tag, current_tag) in connects:
#            prev_resid, current_resid = connects[(prev_tag, current_tag)]
#          else:
#            msg=("Trying to connect residue {} with residue {} but"
#                 "can find a connect record. Please provide all connect"
#                 "records when using nested macros.")
#            raise IOError(msg)

#          n_nodes = len(expanded_graph.nodes)
#          n_nodes_current = len(macro)
#          n_nodes_prev = len(prev_macro)
#          edge = (n_nodes-n_nodes_macro+current_resid, n_nodes)
#          expanded_graph.add_edge(n_nodes-n_nodes_macro+current_resid,
#                                  n_nodes-n_nodes_macro-n_nodes_prev+prev_resid)

#       else:
#          expanded_graph.add_node(node, attrs=graph.nodes[node])


class MacroString():
    """
    Define a (random) tree graph based on a string.
    The graph is generated each time the `gen_graph`
    attribute is run. For random graph they will be
    independent.
    """


    def __init__(self, string):
        """
        Convert string into definitions of the
        random tree graph.

        Parameters:
        -----------
        string: str
           the input string containing all definitions
        """
        input_params = string.split(":")
        self.name = input_params[0]
        self.levels = int(input_params[1])
        self.bfact = int(input_params[2])
        self.residues = []
        self.weights = []

        for res_prob in input_params[3].split(','):
            name, prob = res_prob.split("-")
            self.residues.append(name)
            self.weights.append(float(prob))

    def gen_graph(self, seed=None):
        """
        Generate a graph from the defnitions stored in this
        instance.
        """
        graph = _branched_graph("dum", self.bfact, self.levels)
        graph = _random_replace_nodes(graph, self.residues, self.weights)
        return graph

def _add_edges(graph, edges, idx, jdx):
    """
    Add edges to a graph using the edge format
    'node_1,node_2', where the node keys refer
    to the node relative to the nodes with the same
    attribute `seqid`. The sequence ids are provided by
    `idx` and `jdx`.

    Parameters
    ----------
    graph: `:class:networkx.Graph`
    edges:  list[str]
        str has the format `node_keyA,node_keyB-node_keyC,node_keyD`
    idx: int
    jdx: int

    Returns
    --------
    `:class:networkx.Graph`
    """
    for edge in edges.split(","):
        node_idx, node_jdx = edge.split("-")
        idx_nodes = find_atoms(graph, "seqid", idx)
        jdx_nodes = find_atoms(graph, "seqid", jdx)

        if len(idx_nodes) == 0:
           msg=("Trying to add connect between block with seqid {} and block with"
                "seqid {}. However, cannot find block with seqid {}.")
           raise IOError(msg.format(idx, jdx, idx))
        elif len(jdx_nodes) == 0:
           msg=("Trying to add connect between block with seqid {} and block with"
                "seqid {}. However, cannot find block with seqid {}.")
           raise IOError(msg.format(idx, jdx, jdx))

        try:
           node_i = idx_nodes[int(node_idx)]
        except IndexError:
           msg=("Trying to add connect between block with seqid {} and block with"
                "seqid {}. However, cannot find resid {} in block with seqid {}.")
           raise IOError(msg.format(idx, jdx, node_idx, idx))

        try:
           node_j = jdx_nodes[int(node_jdx)]
        except IndexError:
           msg=("Trying to add connect between block with seqid {} and block with"
                "seqid {}. However, cannot find resid {} in block with seqid {}.")
           raise IOError(msg.format(idx, jdx, node_idx, idx))

        graph.add_edge(node_i, node_j)

    return graph


def generate_seq_graph(sequence, macros, connects):
    """
    Given a sequence definition in terms of blocks in macros
    generate a graph and add the edges provided in connects.

    Parameters:
    ----------
    sequence: list
      a list of blocks defining a sequence
    macros: dict[str, networkx.graph]
      a dictionary of graphs defining a macro block
    connects: list[str]
      a list of connects

    Returns:
    -------
    `:class:networkx.graph`
    """
    seq_graph = nx.Graph()
    for idx, macro_name in enumerate(sequence):

        if hasattr(macros[macro_name], "gen_graph"):
           sub_graph = macros[macro_name].gen_graph()
        else:
           sub_graph = macros[macro_name]

        nx.set_node_attributes(
            sub_graph, {node: idx for node in sub_graph.nodes}, "seqid")
        seq_graph = nx.disjoint_union(seq_graph, sub_graph)

    if connects:
        for connect in connects:
            idx, jdx, edges = connect.split(":")
            _add_edges(seq_graph, edges, int(idx), int(jdx))

    return seq_graph


def gen_seq(args):

    macros = {}

    if args.from_file:
       force_field = load_library("seq", args.lib, args.inpath)
       for tag_name in args.from_file:
           tag, name = tag_name.split(":")
           macros[tag] = force_field.blocks[name]

    if args.macros:
       for macro_string in args.macros:
           macro = MacroString(macro_string)
           macros[macro.name] = macro

    seq_graph = generate_seq_graph(args.seq, macros, args.connects)

    g_json = json_graph.node_link_data(seq_graph)
    with open(args.outpath, "w") as file_handle:
        json.dump(g_json, file_handle, indent=2)
