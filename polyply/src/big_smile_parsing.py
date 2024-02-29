from collections import defaultdict
import re
import numpy as np
try:
    import pysmiles
except ImportError:
    msg = ("You are using a functionality that requires "
           "the pysmiles package. Use pip install pysmiles ")
    raise ImportError(msg)
import networkx as nx
from vermouth.forcefield import ForceField
from vermouth.molecule import Block
from polyply.src.meta_molecule import MetaMolecule

PATTERNS = {"bond_anchor": "\[\$.*?\]",
            "place_holder": "\[\#.*?\]",
            "annotation": "\|.*?\|",
            "fragment": r'#(\w+)=((?:\[.*?\]|[^,\[\]]+)*)',
            "seq_pattern": r'\{([^}]*)\}(?:\.\{([^}]*)\})?'}

def _find_next_character(string, chars, start):
    for idx, token in enumerate(string[start:]):
        if token in chars:
            return idx+start
    return np.inf

def res_pattern_to_meta_mol(pattern):
    """
    Generate a :class:`polyply.MetaMolecule` from a
    pattern string describing a residue graph with the
    simplified big-smile syntax.

    The syntax scheme consists of two curly braces
    enclosing the residue graph sequence. It can contain
    any enumeration of residues by writing them as if they
    were smile atoms but the atomname is given by # + resname.
    This input fomat can handle branching as well ,however,
    macrocycles are currently not supported.

    General Pattern
    '{' + [#resname_1][#resname_2]... + '}'

    In addition to plain enumeration any residue may be
    followed by a '|' and an integern number that
    specifies how many times the given residue should
    be added within a sequence. For example, a pentamer
    of PEO can be written as:

    {[#PEO][#PEO][#PEO][#PEO][#PEO]}

    or

    {[#PEO]|5}

    The block syntax also applies to branches. Here the convetion
    is that the complete branch including it's first anchoring
    residue is repeated. For example, to generate a PMA-g-PEG
    polymer the following syntax is permitted:

    {[#PMA]([#PEO][#PEO])|5}

    Parameters
    ----------
    pattern: str
        a string describing the meta-molecule

    Returns
    -------
    :class:`polyply.MetaMolecule`
    """
    meta_mol = MetaMolecule()
    current = 0
    branch_anchor = 0
    prev_node = None
    branching = False
    for match in re.finditer(PATTERNS['place_holder'], pattern):
        start, stop = match.span()
        # new branch here
        if pattern[start-1] == '(':
            branching = True
            branch_anchor = prev_node
            recipie = [(meta_mol.nodes[prev_node]['resname'], 1)]
        if stop < len(pattern) and pattern[stop] == '|':
            eon = _find_next_character(pattern, ['[', ')', '(', '}'], stop)
            n_mon = int(pattern[stop+1:eon])
        else:
            n_mon = 1

        resname = match.group(0)[2:-1]
        # collect all residues in branch
        if branching:
            recipie.append((resname, n_mon))

        # add the new residue
        connection = []
        for _ in range(0, n_mon):
            if prev_node is not None:
                connection = [(prev_node, current)]
            meta_mol.add_monomer(current,
                                 resname,
                                 connection)
            prev_node = current
            current += 1

        # terminate branch and jump back to anchor
        branch_stop = _find_next_character(pattern, ['['], stop) >\
                      _find_next_character(pattern, [')'], stop)
        if stop <= len(pattern) and branch_stop and branching:
            branching = False
            prev_node = branch_anchor
            # we have to multiply the branch n-times
            eon_a = _find_next_character(pattern, [')'], stop)
            if stop+1 < len(pattern) and pattern[eon_a+1] == "|":
                eon_b = _find_next_character(pattern, ['[', ')', '(', '}'], eon_a+1)
                # -1 because one branch has already been added at this point
                for _ in range(0,int(pattern[eon_a+2:eon_b])-1):
                    for bdx, (resname, n_mon) in enumerate(recipie):
                        if bdx == 0:
                            anchor = current
                        for _ in range(0, n_mon):
                            connection = [(prev_node, current)]
                            meta_mol.add_monomer(current,
                                                 resname,
                                                 connection)
                            prev_node = current
                            current += 1
                    prev_node = anchor
    return meta_mol

def _big_smile_iter(smile):
    for token in smile:
        yield token

def tokenize_big_smile(big_smile):
    """
    Processes a BigSmile string by storing the
    the BigSmile specific bonding descriptors
    in a dict with refernce to the atom they
    refer to. Furthermore, a cleaned smile
    string is generated with the BigSmile
    specific syntax removed.

    Parameters
    ----------
    smile: str
        a BigSmile smile string

    Returns
    -------
    str
        a canonical smile string
    dict
        a dict mapping bonding descriptors
        to the nodes within the smile
    """
    smile_iter = _big_smile_iter(big_smile)
    bonding_descrpt = defaultdict(list)
    smile = ""
    node_count = 0
    prev_node = 0
    for token in smile_iter:
        if token == '[':
            peek = next(smile_iter)
            if peek in ['$', '>', '<']:
                bond_descrp = peek
                peek = next(smile_iter)
                while peek != ']':
                    bond_descrp += peek
                    peek = next(smile_iter)
                bonding_descrpt[prev_node].append(bond_descrp)
            else:
                smile = smile + token + peek
                prev_node = node_count
                node_count += 1

        elif token == '(':
            anchor = prev_node
            smile += token
        elif token == ')':
            prev_node = anchor
            smile += token
        else:
            if token not in '] H @ . - = # $ : / \\ + - %'\
                and not token.isdigit():
                prev_node = node_count
                node_count += 1
            smile += token
    return smile, bonding_descrpt

def _rebuild_h_atoms(mol_graph):
    # special hack around to fix
    # pysmiles bug for a single
    # atom molecule; we assume that the
    # hcount is just wrong and set it to
    # the valance number minus bonds minus
    # bonding connectors
    if len(mol_graph.nodes) == 1:
        ele = mol_graph.nodes[0]['element']
        # for N and P we assume the regular valency
        hcount = pysmiles.smiles_helper.VALENCES[ele][0]
        if mol_graph.nodes[0].get('bonding', False):
            hcount -= 1
        mol_graph.nodes[0]['hcount'] = hcount
    else:
        for node in mol_graph.nodes:
            if mol_graph.nodes[node].get('bonding', False):
                hcount = mol_graph.nodes[node]['hcount']
                mol_graph.nodes[node]['hcount'] = hcount - len(mol_graph.nodes[node]['bonding'])

    pysmiles.smiles_helper.add_explicit_hydrogens(mol_graph)
    return mol_graph

def fragment_iter(fragment_str):
    """
    Iterates over fragments defined in a BigSmile string.
    Fragments are named residues that consist of a single
    smile string together with the BigSmile specific bonding
    descriptors. The function returns the resname of a named
    fragment as well as a plain nx.Graph of the molecule
    described by the smile. Bonding descriptors are annotated
    as node attributes with the keyword bonding.

    Parameters
    ----------
    fragment_str: str
        the string describing the fragments

    Yields
    ------
    str, nx.Graph
    """
    for fragment in fragment_str[1:-1].split(','):
        delim = fragment.find('=', 0)
        resname = fragment[1:delim]
        big_smile = fragment[delim+1:]
        smile, bonding_descrpt = tokenize_big_smile(big_smile)

        if smile == "H":
            mol_graph = nx.Graph()
            mol_graph.add_node(0, element="H", bonding=bonding_descrpt[0])
            nx.set_node_attributes(mol_graph, bonding_descrpt, 'bonding')
        else:
            mol_graph = pysmiles.read_smiles(smile)
            nx.set_node_attributes(mol_graph, bonding_descrpt, 'bonding')
            # we need to rebuild hydrogen atoms now
            _rebuild_h_atoms(mol_graph)

        atomnames = {node[0]: node[1]['element']+str(node[0]) for node in mol_graph.nodes(data=True)}
        nx.set_node_attributes(mol_graph, atomnames, 'atomname')
        nx.set_node_attributes(mol_graph, resname, 'resname')
        yield resname, mol_graph

def force_field_from_fragments(fragment_str):
    """
    Collects the fragments defined in a BigSmile string
    as :class:`vermouth.molecule.Blocks` in a force-field
    object. Bonding descriptors are annotated as node
    attribtues.

    Parameters
    ----------
    fragment_str: str
        string using BigSmile fragment syntax

    Returns
    -------
    :class:`vermouth.forcefield.ForceField`
    """
    force_field = ForceField("big_smile_ff")
    frag_iter = fragment_iter(fragment_str)
    for resname, mol_graph in frag_iter:
        mol_block = Block(mol_graph)
        force_field.blocks[resname] = mol_block
    return force_field
