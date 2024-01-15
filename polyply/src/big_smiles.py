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

def find_token_indices(line, target):
    idxs = [idx for idx, token in enumerate(line) if token == target]
    for idx in idxs:
        yield idx

def compatible(left, right):
    if left == right:
        return True
    if left[0] == "<" and right[0] == ">":
        if left[1:] == right[1:]:
            return True
    if left[0] == ">" and right[0] == "<":
        if left[1:] == right[1:]:
            return True
    return False

def find_compatible_pair(polymol, residue, bond_type="bond_type", eligible_nodes=None):
    ref_nodes = nx.get_node_attributes(polymol, bond_type)
    target_nodes = nx.get_node_attributes(residue, bond_type)
    for ref_node in ref_nodes:
        if eligible_nodes and\
           polymol.nodes[ref_node]['resid'] not in eligible_nodes:
            continue
        for target_node in target_nodes:
            if compatible(ref_nodes[ref_node],
                          target_nodes[target_node]):
                return ref_node, target_node
    return None

class BigSmileParser:

    def __init__(self):
        self.molecule =

    def parse_stochastic_object():


def read_simplified_big_smile_string(line):

    # split the different stochastic objects
    line = line.strip()
    # a stochastic object is enclosed in '{' and '}'
    start_idx = next(find_token_indices(line, "{"))
    stop_idx = next(find_token_indices(line, "}"))
    stoch_line = line[start_idx+1:stop_idx]
    # residues are separated by , and end
    # groups by ;
    if ';' in stoch_line:
        residue_string, terminii_string = stoch_line.split(';')
    else:
        residue_string = stoch_line
        terminii_string = None
    # let's read the smile residue strings
    residues = []
    count = 0
    for residue_string in residue_string.split(','):
        # figure out if this is a named object
        if residue_string[0] == "#":
            jdx = next(find_token_indices(residue_string, "="))
            name = residue_string[:jdx]
            residue_string = residue_string[jdx:]
        else:
            name = count

        mol_graph = read_smiles(residue_string)
        residues.append((name, mol_graph))
        count += 1
    # let's read the terminal residue strings
    end_groups = []
    if terminii_string:
        for terminus_string in terminii_string.split(','):
            mol_graph = read_smiles(terminus_string)
            bond_types = nx.get_node_attributes(mol_graph, "bond_type")
            nx.set_node_attributes(mol_graph, bond_types, "ter_bond_type")
            end_groups.append(mol_graph)
    return cls(dict(residues), end_groups)



