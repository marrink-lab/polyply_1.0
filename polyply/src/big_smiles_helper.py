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
try:
    import pysmiles
except ImportError:
    msg = "The tool you are using requires pysmiles as dependcy."
    raise ImportError(msg)

from pysmiles.read_smiles import _tokenize

def find_anchor(mol, pre_mol, atom):
    anchors = list(pre_mol.neighbors(atom))
    for anchor in anchors:
        if anchor in mol.nodes:
            return False, anchor
    for anchor in nx.ego_graph(pre_mol, atom, radius=2).nodes:
        if anchor in mol.nodes:
            return True, anchor
    raise RuntimeError

def parse_atom(atom):
    """
    Parses a SMILES atom token, and returns a dict with the information.

    Note
    ----
    Can not deal with stereochemical information yet. This gets discarded.

    Parameters
    ----------
    atom : str
        The atom string to interpret. Looks something like one of the
        following: "C", "c", "[13CH3-1:2]"

    Returns
    -------
    dict
        A dictionary containing at least 'element', 'aromatic', and 'charge'. If
        present, will also contain 'hcount', 'isotope', and 'class'.
    """
    defaults = {'charge': 0, 'hcount': 0, 'aromatic': False}
    if atom.startswith('[') and any(mark in atom for mark in ['$', '>', '<']):
        bond_type = atom[1:-1]
        # we have a big smile bond anchor
        defaults.update({"element": None,
                         "bond_type": bond_type})
        return defaults

    if atom.startswith('[') and '#' == atom[1]:
        # this atom is a replacable place holder
        defaults.update({"element": None, "replace": atom[2:-1]})
        return defaults

    if not atom.startswith('[') and not atom.endswith(']'):
        if atom != '*':
            # Don't specify hcount to signal we don't actually know anything
            # about it
            return {'element': atom.capitalize(), 'charge': 0,
                    'aromatic': atom.islower()}
        else:
            return defaults.copy()

    match = ATOM_PATTERN.match(atom)

    if match is None:
        raise ValueError('The atom {} is malformatted'.format(atom))

    out = defaults.copy()
    out.update({k: v for k, v in match.groupdict().items() if v is not None})

    if out.get('element', 'X').islower():
        out['aromatic'] = True

    parse_helpers = {
        'isotope': int,
        'element': str.capitalize,
        'stereo': lambda x: x,
        'hcount': parse_hcount,
        'charge': parse_charge,
        'class': int,
        'aromatic': lambda x: x,
    }

    for attr, val_str in out.items():
        out[attr] = parse_helpers[attr](val_str)

    if out['element'] == '*':
        del out['element']

    if out.get('element') == 'H' and out.get('hcount', 0):
        raise ValueError("A hydrogen atom can't have hydrogens")

    if 'stereo' in out:
        LOGGER.warning('Atom "%s" contains stereochemical information that will be discarded.', atom)

    return out

def big_smile_str_to_graph(smile_str):
    """
    
    """
    bond_to_order = {'-': 1, '=': 2, '#': 3, '$': 4, ':': 1.5, '.': 0}
    pre_mol = nx.Graph()
    anchor = None
    idx = 0
    default_bond = 1
    next_bond = None
    branches = []
    ring_nums = {}
    for tokentype, token in _tokenize(smiles):
        if tokentype == TokenType.ATOM:
            pre_mol.add_node(idx, **parse_atom(token))
            if anchor is not None:
                if next_bond is None:
                    next_bond = default_bond
                if next_bond or zero_order_bonds:
                    pre_mol.add_edge(anchor, idx, order=next_bond)
                next_bond = None
            anchor = idx
            idx += 1
        elif tokentype == TokenType.BRANCH_START:
            branches.append(anchor)
        elif tokentype == TokenType.BRANCH_END:
            anchor = branches.pop()
        elif tokentype == TokenType.BOND_TYPE:
            if next_bond is not None:
                raise ValueError('Previous bond (order {}) not used. '
                                 'Overwritten by "{}"'.format(next_bond, token))
            next_bond = bond_to_order[token]
        elif tokentype == TokenType.RING_NUM:
            if token in ring_nums:
                jdx, order = ring_nums[token]
                if next_bond is None and order is None:
                    next_bond = default_bond
                elif order is None:  # Note that the check is needed,
                    next_bond = next_bond  # But this could be pass.
                elif next_bond is None:
                    next_bond = order
                elif next_bond != order:  # Both are not None
                    raise ValueError('Conflicting bond orders for ring '
                                     'between indices {}'.format(token))
                # idx is the index of the *next* atom we're adding. So: -1.
                if pre_mol.has_edge(idx-1, jdx):
                    raise ValueError('Edge specified by marker {} already '
                                     'exists'.format(token))
                if idx-1 == jdx:
                    raise ValueError('Marker {} specifies a bond between an '
                                     'atom and itself'.format(token))
                if next_bond or zero_order_bonds:
                    pre_mol.add_edge(idx - 1, jdx, order=next_bond)
                next_bond = None
                del ring_nums[token]
            else:
                if idx == 0:
                    raise ValueError("Can't have a marker ({}) before an atom"
                                     "".format(token))
                # idx is the index of the *next* atom we're adding. So: -1.
                ring_nums[token] = (idx - 1, next_bond)
                next_bond = None
        elif tokentype == TokenType.EZSTEREO:
            LOGGER.warning('E/Z stereochemical information, which is specified by "%s", will be discarded', token)
    if ring_nums:
        raise KeyError('Unmatched ring indices {}'.format(list(ring_nums.keys())))

    return pre_mol

def mol_graph_from_big_smile_graph(pre_mol):
    # here we condense any BigSmilesBonding information
    clean_nodes = [node for node in pre_mol.nodes(data=True) if 'bond_type' not in node[1]]
    mol = nx.Graph()
    mol.add_nodes_from(clean_nodes)
    mol.add_edges_from([edge for edge in pre_mol.edges if edge[0] in mol.nodes and edge[1] in mol.nodes])
    for node in pre_mol.nodes:
        if 'bond_type' in pre_mol.nodes[node]:
            terminus, anchor = find_anchor(mol, pre_mol, node)
            if terminus:
                mol.nodes[anchor].update({"ter_bond_type": pre_mol.nodes[node]['bond_type'],
                                          "ter_bond_probs": pre_mol.nodes[node]['bond_probs']})
            else:
                mol.nodes[anchor].update({"bond_type": pre_mol.nodes[node]['bond_type'],
                                          "bond_probs": pre_mol.nodes[node]['bond_probs']})
    return mol
