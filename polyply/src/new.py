import re

PATTERNS = {"bond_anchor": "\[\$.*?\]",
            "place_holder": "\[\#.*?\]",
            "annotation": "\|.*?\|",
            "fragment": r'#(\w+)=((?:\[.*?\]|[^,\[\]]+)*)',
            "seq_pattern": r'\{([^}]*)\}(?:\.\{([^}]*)\})?'}

def read_big_smile(line):
    res_graphs = []
    seq_str, patterns = re.findall(PATTERNS['seq_pattern'], line)[0]
    fragments = dict(re.findall(PATTERNS['fragment'], patterns))
    for fragment in fragments:
        res_graphs.append(read_smile_w_bondtypes(fragment_smile))

    # now stitch together ..
    # 1 segement the seq_str
    # allocate any leftover atoms
    # add the residues
    targets = set()
    for match in re.finditer(PATTERNS['place_holder'], seq_str):
       targets.add(match.group(0))
    for target in targets:
       seq_str = seq_str.replace(target, fragments[target[2:-1]])
       
    return seq_str

def read_smile_w_bondtypes(line):
    smile = line
    bonds=[]
    # find all bond types and remove them from smile
    for bond in re.finditer(PATTERNS['bond_anchor'], ex_str):
        smile=smile.replace(bond.group(0), "")
        bonds.append((bond.span(0), bond.group(0)[1:-1]))

    # read smile and make molecule
    mol = read_smiles(smile)
    pos_to_node = position_to_node(smile)

    # strip the first terminal anchor if there is any //

    # associate the bond atoms with the smile atoms
    for bond in bonds:
        # the bondtype contains the zero index so it
        # referes to the first smile node
        if bond[0][0] == 0:
            mol.nodes[0]['bondtype'] = bond[1]
        else:
            anchor = find_anchor(smile, bond[0][0])
            mol.nodes[anchor]['bondtype'] = bond[1]

    return mol


def find_anchor(smile, start):
    branch = False
    sub_smile=smile[:start]
    for idx, token in enumerate(sub_smile[::-1]):
        if token == ")":
            branch = True
            continue
        if token == "(" and branch:
            branch = False
            continue
        if not branch:
            return start-idx
    raise IndexError

def position_to_node(smile):
    count=0
    pos_to_node={}
    for idx, token in enumerate(smile):
        if token not in ['[', ']', '$', '@', '(', ')']:
            pos_to_node[idx] = count
            count+=1
    return pos_to_node
