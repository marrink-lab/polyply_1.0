# Copyright 2024 Dr. Fabian Gruenewald
#
# Licensed under the PolyForm Noncommercial License 1.0.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://polyformproject.org/licenses/noncommercial/1.0.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import networkx as nx
from vermouth.forcefield import ForceField
from vermouth.gmx.itp_read import read_itp
from polyply.src.meta_molecule import MetaMolecule
from polyply.src.topology import Topology
from polyply.src.molecule_utils import extract_block, extract_links, find_termini_mods, handle_ptms
from polyply.src.fragment_finder import FragmentFinder
from polyply.src.ffoutput import ForceFieldDirectiveWriter
from .load_library import load_ff_library

def is_opls(topology):
    atomtypes = list(topology.atom_types.keys())
    if "opls" in atomtypes[0]:
        return True
    return False

def _clean_opls_atomtypes(topology):
    old_to_new = {}
    unique_atypes = {}

    for atype, params in topology.atom_types.items():
        nb_vals = (str(params['nb1']), str(params['nb2']))
        if nb_vals not in unique_atypes:
            unique_atypes[nb_vals] = atype
        old_to_new[atype] = unique_atypes[nb_vals]
    for mol in topology.molecules:
        for node in mol.molecule.nodes:
            mol.molecule.nodes[node]["atype"] = old_to_new[mol.molecule.nodes[node]["atype"]]
        mol.relabel_and_redo_res_graph(mapping={})
    return topology

def _make_edges_from_vs(molecule):
    for inter_type, inters in molecule.interactions.items():
        if "virtual" in inter_type:
            for inter in inters:
                ref = inter.atoms[0]
                for anchor in inter.atoms[1:]:
                    molecule.add_edge(ref, anchor)

def _read_itp_file(itppath):
    """
    small wrapper for reading itps
    """
    with open(itppath, "r") as _file:
        lines = _file.readlines()
    force_field = ForceField("tmp")
    read_itp(lines, force_field)
    block = next(iter(force_field.blocks.values()))
    mol = block.to_molecule()
    mol.make_edges_from_interaction_type(type_="bonds")
    # make edges from VS
    _make_edges_from_vs(mol)
    return mol

def gen_ff(itppath, smile_str, outpath, inpath=[], res_charges=None):
    """
    Main executable for gen_ff tool.
    """
    # load FF files if given
    if inpath:
        force_field = load_ff_library("new", None, inpath)
    # if none are given we create an empty ff
    else:
        force_field = ForceField("new")

    # what charges belong to which resname
    if res_charges:
        crg_dict = dict(res_charges)

    # read the topology file
    if itppath.suffix == ".top":
        top = Topology.from_gmx_topfile(itppath, name="test")
        # opls specific fix
        # in LigParGen each atom get's its own atype even though
        # they are the same; pretty strange but this confuses
        # the terminal modifications module
        if top and is_opls(top):
            _clean_opls_atomtypes(top)
        target_mol = top.molecules[0].molecule
    # read itp file
    elif itppath.suffix == ".itp":
        top = None
        target_mol = _read_itp_file(itppath)

    # read the big-smile representation
    meta_mol = MetaMolecule.from_cgsmiles_str(force_field=force_field,
                                          mol_name="ref",
                                          cgsmiles_str=smile_str,
                                          seq_only=False,
                                          all_atom=True)

    # identify and extract all unique fragments
    unique_fragments, res_graph = FragmentFinder(target_mol).extract_unique_fragments(meta_mol.molecule)

    # group fragments by resname and assign ptm modifications if
    # they are not subgraph isomorphic
    modification_names = handle_ptms(top, unique_fragments, res_graph,
                                     target_mol, force_field, crg_dict)

    # extract the regular links
    force_field.links += extract_links(target_mol)
    # extract links that span the terminii; they also annotate the
    # terminal residues with the modifications generated above
    find_termini_mods(res_graph, target_mol, force_field, modification_names)

    with open(outpath, "w") as filehandle:
        ForceFieldDirectiveWriter(forcefield=force_field, stream=filehandle).write()
