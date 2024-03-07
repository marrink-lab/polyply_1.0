from collections import defaultdict
import re
import numpy as np
try:
    import pysmiles
except ImportError as error:
    msg = ("You are using a functionality that requires "
           "the pysmiles package. Use pip install pysmiles ")
    raise ImportError(msg) from error
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

def _expand_branch(meta_mol, current, anchor, recipe):
    prev_node = anchor
    for bdx, (resname, n_mon) in enumerate(recipe):
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
    return meta_mol, current, prev_node

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
    followed by a '|' and an integer number that
    specifies how many times the given residue should
    be added within a sequence. For example, a pentamer
    of PEO can be written as:

    {[#PEO][#PEO][#PEO][#PEO][#PEO]}

    or

    {[#PEO]|5}

    The block syntax also applies to branches. Here the convention
    is that the complete branch including it's first anchoring
    residue is repeated. For example, to generate a PMA-g-PEG
    polymer containing 15 residues the following syntax is permitted:

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
    # stores one or more branch anchors; each next
    # anchor belongs to a nested branch
    branch_anchor = []
    # used for storing composition protocol for
    # for branches; each entry is a list of
    # branches from extending from the anchor
    # point
    recipes = defaultdict(list)
    # the previous node
    prev_node = None
    # do we have an open branch
    branching = False
    # each element in the for loop matches a pattern
    # '[' + '#' + some alphanumeric name + ']'
    for match in re.finditer(PATTERNS['place_holder'], pattern):
        start, stop = match.span()
        # we start a new branch when the residue is preceded by '('
        # as in ... ([#PEO] ...
        if pattern[start-1] == '(':
            branching = True
            branch_anchor.append(prev_node)
            # the recipe for making the branch includes the anchor; which
            # is hence the first residue in the list
            recipes[branch_anchor[-1]] = [(meta_mol.nodes[prev_node]['resname'], 1)]
        # here we check if the atom is followed by a expansion character '|'
        # as in ... [#PEO]|
        if stop < len(pattern) and pattern[stop] == '|':
            # eon => end of next
            # we find the next character that starts a new residue, ends a branch or
            # ends the complete pattern
            eon = _find_next_character(pattern, ['[', ')', '(', '}'], stop)
            # between the expansion character and the eon character
            # is any number that correspnds to the number of times (i.e. monomers)
            # that this atom should be added
            n_mon = int(pattern[stop+1:eon])
        else:
            n_mon = 1

        # the resname starts at the second character and ends
        # one before the last according to the above pattern
        resname = match.group(0)[2:-1]
        # if this residue is part of a branch we store it in
        # the recipe dict together with the anchor residue
        # and expansion number
        if branching:
            recipes[branch_anchor[-1]].append((resname, n_mon))

        # new we add new residue as often as required
        connection = []
        for _ in range(0, n_mon):
            if prev_node is not None:
                connection = [(prev_node, current)]
            meta_mol.add_monomer(current,
                                 resname,
                                 connection)
            prev_node = current
            current += 1

        # here we check if the residue considered before is the
        # last residue of a branch (i.e. '...[#residue])'
        # that is the case if the branch closure comes before
        # any new atom begins
        branch_stop = _find_next_character(pattern, ['['], stop) >\
                      _find_next_character(pattern, [')'], stop)

        # if the branch ends we reset the anchor
        # and set branching False unless we are in
        # a nested branch
        if stop <= len(pattern) and branch_stop:
            branching = False
            prev_node = branch_anchor.pop()
            if branch_anchor:
                branching = True
            #========================================
            #       expansion for branches
            #========================================
            # We need to know how often the branch has
            # to be added so we first identify the branch
            # terminal character ')' called eon_a.
            eon_a = _find_next_character(pattern, [')'], stop)
            # Then we check if the expansion character
            # is next.
            if stop+1 < len(pattern) and pattern[eon_a+1] == "|":
                # If there is one we find the beginning
                # of the next branch, residue or end of the string
                # As before all characters inbetween are a number that
                # is how often the branch is expanded.
                eon_b = _find_next_character(pattern, ['[', ')', '(', '}'], eon_a+1)
                # the outermost loop goes over how often a the branch has to be
                # added to the existing sequence
                for idx in range(0,int(pattern[eon_a+2:eon_b])-1):
                    prev_anchor = None
                    skip = 0
                    # in principle each branch can contain any number of nested branches
                    # each branch is itself a recipe that has an anchor atom
                    for ref_anchor, recipe in list(recipes.items())[len(branch_anchor):]:
                        # starting from the first nested branch we have to do some
                        # math to find the anchor atom relative to the first branch
                        # we also skip the first residue in recipe, which is the
                        # anchor residue. Only the outermost branch in an expansion
                        # is expanded including the anchor. This allows easy description
                        # of graft polymers.
                        if prev_anchor:
                            offset = ref_anchor - prev_anchor
                            prev_node = prev_node + offset
                            skip = 1
                        # this function simply adds the residues of the paticular
                        # branch
                        meta_mol, current, prev_node = _expand_branch(meta_mol,
                                                                      current=current,
                                                                      anchor=prev_node,
                                                                      recipe=recipe[skip:])
                        # if this is the first branch we want to set the anchor
                        # as the base anchor to which we jump back after all nested
                        # branches have been added
                        if prev_anchor is None:
                            base_anchor = prev_node
                        # store the previous anchor so we can do the math for nested
                        # branches
                        prev_anchor = ref_anchor
                # all branches added; then go back to the base anchor
                prev_node = base_anchor
            # if all branches are done we need to reset the lists
            # when all nested branches are completed
            if len(branch_anchor) == 0:
                recipes = defaultdict(list)
    return meta_mol

def tokenize_big_smile(big_smile):
    """
    Processes a BigSmile string by storing the
    the BigSmile specific bonding descriptors
    in a dict with reference to the atom they
    refer to. Furthermore, a cleaned smile
    string is generated with the BigSmile
    specific syntax removed.

    Parameters
    ----------
    smile: str
        a BigSmile smiles string

    Returns
    -------
    str
        a canonical smiles string
    dict
        a dict mapping bonding descriptors
        to the nodes within the smiles string
    """
    smile_iter = iter(big_smile)
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
                # get the degree
                ele = mol_graph.nodes[0]['element']
                # hcoung is the valance minus the degree minus
                # the number of bonding descriptors
                hcount = pysmiles.smiles_helper.VALENCES[ele][0] -\
                         mol_graph.degree(node) -\
                         len(mol_graph.nodes[node]['bonding'])

                mol_graph.nodes[node]['hcount'] = hcount

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

# ToDos
# - remove special case hydrogen line 327ff
# - check rebuild_h and clean up
