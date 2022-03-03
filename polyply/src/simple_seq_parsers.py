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
import networkx as nx
from vermouth.parser_utils import split_comments

ONE_LETTER_AAS = {
 "A": "ALA",
 "R": "ARG",
 "N": "ASN",
 "D": "ASP",
 "C": "CYS",
 "Q": "GLB",
 "E": "GLU",
 "G": "GLY",
 "H": "HIS",
 "I": "ILE",
 "L": "LEU",
 "K": "LYS",
 "M": "MET",
 "F": "PHE",
 "P": "PRO",
 "O": "PLY",
 "S": "SER",
 "U": "SEC",
 "T": "THR",
 "W": "TRP",
 "Y": "TYR",
 "V": "VAL",
 "B": "ASX",
 "Z": "GLX",
 "X": "XAA",
 "J": "XLE",}

ONE_LETTER_DNA = {"A": "DA",
                  "C": "DC",
                  "G": "DG",
                  "T": "DT"}

ONE_LETTER_RNA = {"A": "A",
                  "C": "C",
                  "G": "G",
                  "T": "U"}

def _monomers_to_linear_nx_graph(monomers):
    seq_graph = nx.Graph()
    seq_graph.add_edges_from(zip(range(0, len(monomers))))
    nx.add_node_attributes(seq_graph, {node: resname for node, resname in zip(seq_graph.nodes, monomers)}, "resname")
    nx.add_node_attributes(seq_graph, {node: node+1 for node in zip(seq_graph.nodes)}, "resid")
    return seq_graph

def parse_plain_delimited(lines, delimiter=" "):
    """
    Parse a plain delimited text file. The delimiter can
    be any special character.
    """
    monomers = []
    for line in lines:
        for resname in line.strip().split(delimiter):
            monomers.append(resname)
    seq_graph =  _monomers_to_linear_nx_graph(monomers)
    return seq_graph

def parse_plain(lines, DNA=False, RNA=False):
    """
    Parse a plain one letter sequence block either for DNA, RNA,
    or amino-acids. Lines can be a list of strings or a string.
    For the format see here:

    https://www.animalgenome.org/bioinfo/resources/manuals/seqformats
    """
    monomers = []
    for line in lines:
        for token in line.strip():
            if token in ONE_LETTER_AAS and not DNA and not RNA:
                resname = ONE_LETTER_AAS[token]
            elif token in ONE_LETTER_DNA and DNA:
                resname = ONE_LETTER_DNA[token]
            elif token in ONE_LETTER_RNA and RNA:
                resname = ONE_LETTER_RNA[token]
            else:
                msg = "Cannot find one letter residue match for { }"
                raise IOError(msg)

            monomers.append(resname)

    seq_graph =  _monomers_to_linear_nx_graph(monomers)
    return seq_graph

def parse_ig(lines):
    """
    Read ig sequence in DNA/AA formatted file. See following link
    for format:

    https://www.animalgenome.org/bioinfo/resources/manuals/seqformats
    """
    clean_lines = []
    idx = 0
    for idx, line in enumerate(lines):
        clean_line, _ = split_comments(line)

        if clean_line[:-1] == '1' or clean_line[:-1] == '2':
            ter_char = clean_line[:-1]
            clean_line = clean_line[:-2]
            clean_lines.append(clean_line)
            seq_graph = parse_plain(clean_lines)
            if ter_char == '2':
                seq_graph.add_edge(0, len(seq_graph.nodes))
            break

    if idx < len(lines):
        print("Warning found mroe than one sequence. Only taking the first one")

    return seq_graph

def parse_fasta(lines):
    """
    Read ig sequence in DNA/AA formatted file. See following link
    for format:

    https://www.animalgenome.org/bioinfo/resources/manuals/seqformats
    """
    clean_lines = []
    count = 0
    for line in lines:
        if '>' in line:
            count += 1

        if count > 1:
            print("Found more than 1 sequence. I am only using the first one.")
            break

        clean_line, _ = split_comments(line, comment_char='>')
        clean_lines.append(clean_line)

    seq_graph = parse_plain(clean_lines)
    return seq_graph
