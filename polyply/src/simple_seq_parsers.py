# Copyright 2022 University of Groningen
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
from collections import OrderedDict
from functools import partial
import json
import networkx as nx
from networkx.readwrite import json_graph
from vermouth.parser_utils import split_comments

ONE_LETTER_DNA = {"A": "DA",
                  "C": "DC",
                  "G": "DG",
                  "T": "DT"}

ONE_LETTER_RNA = {"A": "A",
                  "C": "C",
                  "G": "G",
                  "T": "U"}

def _monomers_to_linear_nx_graph(monomers):
    """
    Take a list of monomers and generate a linear
    graph from them setting both the resid and
    resname attribute.
    """
    seq_graph = nx.Graph()
    mon_range = range(0, len(monomers))
    seq_graph.add_edges_from(zip(mon_range[:-1], mon_range[1:]))
    nx.set_node_attributes(seq_graph, {node: resname for node, resname in zip(seq_graph.nodes, monomers)}, "resname")
    nx.set_node_attributes(seq_graph, {node: node+1 for node in seq_graph.nodes}, "resid")
    return seq_graph

def parse_plain_delimited(filehandle, delimiter=" "):
    """
    Parse a plain delimited text file. The delimiter can
    be any special character.
    """
    with open(filehandle) as file_:
        lines = file_.readlines()

    monomers = []
    for line in lines:
        for resname in line.strip().split(delimiter):
            monomers.append(resname)
    seq_graph =  _monomers_to_linear_nx_graph(monomers)
    return seq_graph

parse_csv = partial(parse_plain_delimited, delimiter=',')
parse_txt = parse_plain_delimited

def parse_plain(lines, DNA=False, RNA=False):
    """
    Parse a plain one letter sequence block either for DNA, RNA,
    or amino-acids. Lines can be a list of strings or a string.
    This function also sets the appropiate defaults for the termini
    of DNA and RNA

    For the format see here:

    https://www.animalgenome.org/bioinfo/resources/manuals/seqformats

    Parameters
    ----------
    lines: `abc.iteratable`
        list of strings matching one letter code DNA, RNA, or AAs
    DNA: bool
        if the sequence matches DNA
    RNA: bool
        if the sequence matches RNA

    Returns
    -------
    `:class:nx.Graph`
        the plain residue graph matching the sequence
    """
    monomers = []
    for line in lines:
        for token in line.strip():
            if token in ONE_LETTER_DNA and DNA:
                resname = ONE_LETTER_DNA[token]
            elif token in ONE_LETTER_RNA and RNA:
                resname = ONE_LETTER_RNA[token]
            else:
                msg = f"Cannot find one letter residue match for {token}"
                raise IOError(msg)

            monomers.append(resname)

    # make sure to set the defaults for the DNA terminals
    if DNA:
        monomers[0] = monomers[0] + "5"
        monomers[-1] = monomers[-1] + "3"

    seq_graph =  _monomers_to_linear_nx_graph(monomers)
    return seq_graph

def _identify_nucleotypes(comments):
    """
    From a comment found in the ig or fasta file, identify if
    the sequence is RNA or DNA sequence by checking if these
    keywords are in the comment lines. Raise an error if
    none or conflicting information are found.

    Parameters
    ----------
    comments: abc.itertable
        list of comment lines

    Returns
    -------
    bool, bool
        is it DNA, RNA

    Raises
    ------
    IOError
        neither RNA nor DNA keywords are found
        both RNA and DNA are found
    """
    RNA = False
    DNA = False
    for comment in comments:
        if "DNA" in comment:
            DNA = True

        if "RNA" in comment:
            print("go here")
            RNA = True

    if RNA and DNA:
        raise IOError("Found both RNA and DNA keyword in comment. Choose one.")

    if not RNA and not DNA:
        raise IOError("Cannot identify if sequence is RNA or DNA from comment.")

    return DNA, RNA

def parse_ig(filehandle):
    """
    Read ig sequence in DNA/AA formatted file. See following link
    for format:

    https://www.animalgenome.org/bioinfo/resources/manuals/seqformats

    Parameters
    ----------
    filehandle: :class:`pathlib.Path`
    """
    with open(filehandle) as file_:
        lines = file_.readlines()

    clean_lines = []
    comments = []
    idx = 0
    for idx, line in enumerate(lines):
        clean_line, comment = split_comments(line)
        comments.append(comment)
        if clean_line:
            if clean_line[-1] == '1' or clean_line[-1] == '2':
                ter_char = clean_line[-1]
                clean_line = clean_line[:-1]
                clean_lines.append(clean_line)
                DNA, RNA = _identify_nucleotypes(comments)
                seq_graph = parse_plain(clean_lines[1:], DNA=DNA, RNA=RNA)

                if ter_char == '2':
                    nnodes = len(seq_graph.nodes)
                    seq_graph.add_edge(0, nnodes-1)
                    seq_graph.edges[(0, nnodes-1)]["circle"] = True
                    seq_graph.nodes[0]["resname"] = seq_graph.nodes[0]["resname"][:-1]
                    seq_graph.nodes[nnodes-1]["resname"] = seq_graph.nodes[nnodes-1]["resname"][:-1]
                break
            else:
                clean_lines.append(clean_line)

    if idx < len(lines):
        print("Warning found mroe than one sequence. Only taking the first one")

    return seq_graph

def parse_fasta(filehandle):
    """
    Read ig sequence in DNA/AA formatted file. See following link
    for format:

    https://www.animalgenome.org/bioinfo/resources/manuals/seqformats
    """
    with open(filehandle) as file_:
        lines = file_.readlines()

    clean_lines = []
    count = 0
    # first line must be a comment line
    DNA, RNA =_identify_nucleotypes([lines[0]])

    for line in lines[1:]:
        if '>' in line:
            print("Found more than 1 sequence. Will only using the first one.")
            break

        clean_lines.append(line)

    seq_graph = parse_plain(clean_lines, RNA=RNA, DNA=DNA)
    return seq_graph

def parse_json(filehandle):
    """
    Read in json file.
    """
    with open(filehandle) as file_:
        data = json.load(file_)

    init_json_graph = nx.Graph(json_graph.node_link_graph(data))
    # the nodes in the inital graph are not ordered, when no resid
    # is given this can create issues. So we reorder the node based
    # on the node key, which HAS to be numerical.
    seq_graph = nx.Graph(node_dict_factory=OrderedDict)
    nodes = list(init_json_graph.nodes(data=True))
    nodes.sort()
    seq_graph.add_nodes_from(nodes)
    seq_graph.add_edges_from(init_json_graph.edges(data=True))
    return seq_graph
