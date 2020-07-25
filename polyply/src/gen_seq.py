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

def _macro_to_graph(resname, branching_f, n_levels):
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

def _add_edges(graph, edges, idx, jdx):
    """
    Add edges to a graph using the edge format
    'node_1,node_2', where the node keys refere
    to the node relative to the nodes with the same
    attribute `seqid`. The sequence ids are provided by
    `idx` and `jdx`.

    Parameters
    ----------
    graph: `:class:networkx.Graph`
    edges:  list[str]
        str has the format `node_key,node_key`
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
        graph.add_edge(idx_nodes[int(node_idx)], jdx_nodes[int(node_jdx)])
    return graph

def _random_macro_to_graph(n_blocks, residues):
    """
    Generate a linear graph with `n_blocks` nodes,
    and set a node attribute according to the probabilites
    defined in `residues`. Residues is a list of strings
    in the form of residue_1-precentage,residue_2-percantage.
    The percantages needs to add up to 1.

    Parameters
    ----------
    n_blocks: int
    residues: list[str]

    Returns:
    --------
    `:class:networkx.Graph`
    """
    macro_graph = _macro_to_graph("dum", 1, int(n_blocks))
    nodes = list(macro_graph.nodes)
    len_graph = len(nodes)
    for res_prob in residues.split(','):
        resname, percentage = res_prob.split('-')
        # this can probably be done smater
        nnodes = int(float(percentage) * len_graph)
        chosen_nodes = random.sample(nodes, k=nnodes)
        res_attr = {node: resname for node in chosen_nodes}
        nx.set_node_attributes(macro_graph, res_attr, "resname")
        for node in chosen_nodes:
            nodes.remove(node)
    return macro_graph

def interpret_macro_string(macro_str, macro_type, force_field=None):
    """
    Given a `macro_str` corresponding to a molecular graph and a
    `macro_type`, generate a graph given the specific requirements.

    Parameters
    ----------
    macro_str:  str
    macro_type: str
         the type of the str; allowed are file, linear, branched,
         random-linear
    force_field: `:class:vermouth.forcefield.ForceField`

    Returns
    -------
    `:class:networkx.Graph`
    """
    if macro_type == "from_file":
        print(macro_str)
        name, mol_name = macro_str.split(":")
        macro = force_field.blocks[mol_name]
    elif macro_type == "linear":
        name, resname, n_blocks = macro_str.split(":")
        macro = _macro_to_graph(resname, 1, int(n_blocks))
    elif macro_type == "branched":
        name, resname, branching_f, n_levels = macro_str.split(":")
        macro = _macro_to_graph(resname, int(branching_f), int(n_levels))
    elif macro_type == "random-linear":
        name, n_blocks, residues = macro_str.split(":")
        macro = _random_macro_to_graph(n_blocks, residues)
    return name, macro

def generate_seq_graph(sequence, macros, connects):
    """
    Given a sequence definition in terms of blocks in macros
    generate a graph and add the edges provided in connects.

    Parameters:
    ----------
    sequence: list
      a list of blocks defining a squence
    macros: dict[name]:networkx.graph
      a dictionary of graphs defining a macro block
    connects:
      a list of connects

    Returns:
    -------
    `:class:networkx.graph`
    """
    seq_graph = nx.Graph()
    for idx, macro_name in enumerate(sequence):
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

    for macro_type in ["from_file", "linear", "branched", "random_linear"]:
        macro_strings = getattr(args, macro_type)

        if macro_strings:
           if macro_type == "from_file" and macro_strings:
               force_field = load_library("seq", args.lib, args.inpath)
           else:
               force_field = None

           for macro_string in macro_strings:
               name, macro_graph = interpret_macro_string(macro_string,
                                                          macro_type,
                                                          force_field)
               macros[name] = macro_graph

    seq_graph = generate_seq_graph(args.seq, macros, args.connects)

    g_json = json_graph.node_link_data(seq_graph)
    with open(args.outpath, "w") as file_handle:
        json.dump(g_json, file_handle, indent=2)
