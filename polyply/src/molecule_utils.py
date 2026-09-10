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
import itertools
from collections import defaultdict
import numpy as np
import networkx as nx
import vermouth
from vermouth.log_helpers import StyleAdapter, get_logger
from vermouth.molecule import Interaction
from polyply.tests.test_lib_files import _interaction_equal
from .topology import replace_defined_interaction
from .graph_utils import find_connecting_edges, find_one_subgraph_match
from .charges import balance_charges, set_charges

LOGGER = StyleAdapter(get_logger(__name__))

def diffs_to_prefix(atoms, resid_diffs):
    """
    Given a list of atoms and corresponding differences
    between their resids, generate the offset prefix for
    the atomnames according to the vermouth sepcific offset
    language.

    The reference atom must have resid_diff value of 0.
    Other atoms either get - or + signs
    depending on their resid offset.

    Parameters
    ----------
    atoms: abc.itertable[str]
    resid_diff: abc.itertable[int]
        the differences in resid with respeect to
        the smallest/largest resid which is 0

    Returns
    -------
    abc.itertable
        list with prefixed atom names
    """
    prefixed_atoms = []
    for atom, diff in zip(atoms, resid_diffs):
        if diff > 0:
            prefix = "".join(["+" for i in range(0, diff)])
        else:
            prefix = "".join(["-" for i in range(diff, 0)])
        prefixed_atoms.append(prefix + atom)
    return prefixed_atoms

def _extract_edges_from_shortest_path(atoms, block, min_resid):
    """
    Given a list atoms generate a list of edges correspoding to
    all edges required to connect all atoms by at least one
    shortest path. Edges are retunred on atomname basis with
    prefix relative to the `min_resid`. See diffs_to_prefix.

    Paramters:
    ----------
    atoms: abc.itertable
        the atoms to collect edges for
    block: :class:`vermouth.molecule.Block`
        the molecule which to servey for edges
    min_resid: int
        the resid to which the prefix indicate relative resid
        distance

    Returns
    -------
    list[tuple]
        the edge list by atomname with prefix indicating relative
        residue distance to min_resid
    """
    edges = []
    had_edges = []
    final_atoms = {}
    resnames = {}
    for origin, target in itertools.combinations(atoms, r=2):
        path = list(nx.shortest_simple_paths(block, source=origin, target=target))[0]
        for edge in zip(path[:-1], path[1:]):
            if edge not in had_edges:
                resid_diffs = np.array([block.nodes[node]['resid'] for node in edge]) - min_resid
                atom_names = [block.nodes[node]["atomname"] for node in edge]
                link_names = diffs_to_prefix(atom_names, resid_diffs)
                final_atoms.update(dict(zip(edge, link_names)))
                edges.append(link_names)
                had_edges.append(edge)
                resnames.update(zip(link_names, [ block.nodes[node]["resname"] for node in edge]))
    return final_atoms, edges, resnames


def extract_links(molecule):
    """
    Given a molecule that has the resid and resname attributes
    correctly set, extract the interactions which span more than
    a single residue and generate a link.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
        the molecule from which to extract interactions

    Returns
    -------
    list[:class:`vermouth.molecule.Links`]
        a list with a links found
    """
    links = []
    # patterns are a sqeuence of atoms that define an interaction
    # sometimes multiple interactions are defined for one pattern
    # in that case they are all collected in this dictionary
    patterns = defaultdict(dict)
    # for each found pattern the resnames are collected; this is important
    # because the same pattern may apply to residues with different name
    resnames_for_patterns = defaultdict(dict)
    link_atoms_for_patterns = defaultdict(list)
    # as additional safe-gaurd against false links we also collect the edges
    # that span the interaction by finding the shortest simple path between
    # all atoms in patterns. Note that the atoms in patterns not always have
    # to be directly bonded. For example, pairs are not directly bonded and
    # can span multiple residues
    for inter_type in molecule.interactions:
        for kdx, interaction in enumerate(molecule.interactions[inter_type]):
            # extract resids and resname corresponding to interaction atoms
            resids = np.array([molecule.nodes[atom]["resid"] for atom in interaction.atoms])
            resnames = [molecule.nodes[atom]["resname"] for atom in interaction.atoms]
            # compute the resid offset to be used for the atom prefixes
            min_resid = min(resids)
            diff = resids - min_resid
            pattern = tuple(set(list(zip(diff, resnames))))
            pattern = tuple(sorted(pattern))
            # in this case all interactions are in a block and we skip
            if np.sum(diff) == 0:
                continue

            # we collect the edges corresponding to the simple paths between pairs of atoms
            # in the interaction
            mol_atoms_to_link_atoms, edges, resnames = _extract_edges_from_shortest_path(interaction.atoms, molecule, min_resid)
            #link_to_mol_atoms = {value:key for key, value in mol_atoms_to_link_atoms.items()}
            link_atoms =  [mol_atoms_to_link_atoms[atom] for atom in interaction.atoms]
            link_inter = Interaction(atoms=link_atoms,
                                     parameters=interaction.parameters,
                                     meta={})

            # here we deal with filtering redundancy
            if pattern in patterns and inter_type in patterns[pattern]:
                for other_inter in patterns[pattern].get(inter_type, []):
                    if _interaction_equal(other_inter, link_inter, inter_type):
                        break
                else:
                    patterns[pattern][inter_type].append(link_inter)
                    resnames_for_patterns[pattern].update(resnames)
                    link_atoms_for_patterns[pattern] += link_atoms
            else:
                patterns[pattern][inter_type] = [link_inter]
                resnames_for_patterns[pattern].update(resnames)
                link_atoms_for_patterns[pattern] += link_atoms
    # we make new links for each unique interaction per type
    for pattern in patterns:
        link = vermouth.molecule.Link()
        link.add_nodes_from(set(link_atoms_for_patterns[pattern]))
        resnames = resnames_for_patterns[pattern]
        nx.set_node_attributes(link, resnames, "resname")

        had_parameters = []
        for inter_type, inters in patterns[pattern].items():
            for idx, interaction in enumerate(inters):
                #new_parameters = interaction.parameters
                new_meta = interaction.meta
                #new_atoms = interaction.atoms
                # to account for the fact when multiple interactions with the same
                # atom patterns need to be written to ff
                if "virtual" not in inter_type and "excl" not in inter_type:
                    # versions are 1-based: apply_links.py treats a missing
                    # 'version' as 1 (i.e. matching/overwriting the base
                    # block interaction), so the first entry here must be
                    # tagged 1, not 0, or it ends up as an extra interaction
                    # alongside the block's instead of replacing it
                    new_meta.update({"version": idx + 1})
                    new_meta.update({"comment": "link"})
                had_parameters.append(interaction.parameters)
                # map atoms to proper atomnames ..
                link.interactions[inter_type].append(interaction)
        links.append(link)
    return links


def _relabel_interaction_atoms(interaction, mapping):
    """
    Relables the atoms in interaction according to the
    rules defined in mapping.

    Parameters
    ----------
    interaction: `vermouth.molecule.Interaction`
    mapping: `:class:dict`

    Returns
    -------
    interaction: `vermouth.molecule.Interaction`
        the new interaction with updated atoms
    """
    new_atoms = [mapping[atom] for atom in interaction.atoms]
    new_interaction = interaction._replace(atoms=new_atoms)
    return new_interaction


def extract_block(molecule, template_graph, defines):
    """
    Given a `vermouth.molecule` and a `resname`
    extract the information of a block from the
    molecule definition and replace all defines
    if any are found.

    Parameters
    ----------
    molecule:  :class:vermouth.molecule.Molecule
    template_graph: :class:`nx.Graph`
        the graph of the template reisdue
    defines:   dict
      dict of type define: value

    Returns
    -------
    :class:vermouth.molecule.Block
    """
    block = vermouth.molecule.Block()

    # select all nodes with the same first resid and
    # make sure the block node labels are atomnames
    # also build a correspondance dict between node
    # label in the molecule and in the block for
    # relabeling the interactions
    mapping = {}
    for node in template_graph.nodes:
        attr_dict = molecule.nodes[node]
        block.add_node(attr_dict["atomname"], **attr_dict)
        mapping[node] = attr_dict["atomname"]

    for inter_type in molecule.interactions:
        had_interactions = []
        versions = {}
        for interaction in molecule.interactions[inter_type]:
            if all(atom in mapping for atom in interaction.atoms):
                interaction = replace_defined_interaction(interaction, defines)
                interaction = _relabel_interaction_atoms(interaction, mapping)
                if tuple(interaction.atoms) in had_interactions:
                    n = versions.get(tuple(interaction.atoms), 1) + 1
                    meta = {"version": n}
                    versions[tuple(interaction.atoms)] = n
                    interaction.meta.update(meta)
                block.interactions[inter_type].append(interaction)
                had_interactions.append(tuple(interaction.atoms))

    for inter_type in ["bonds", "constraints", "virtual_sitesn",
                       "virtual_sites2", "virtual_sites3", "virtual_sites4"]:
        block.make_edges_from_interaction_type(inter_type)

    return block

def _interaction_already_specified(candidate, inter_type, links):
    """
    Check if some link in `links` already has an interaction of `inter_type`
    with the same atoms and parameters as `candidate` (version/comment
    metadata is link-specific bookkeeping and is ignored for this check).

    Parameters
    ----------
    candidate: :class:`vermouth.molecule.Interaction`
    inter_type: str
    links: list[:class:`vermouth.molecule.Link`]

    Returns
    -------
    bool
    """
    candidate = candidate._replace(meta={})
    for link in links:
        for existing in link.interactions.get(inter_type, []):
            if _interaction_equal(candidate, existing._replace(meta={}), inter_type):
                return True
    return False

def _is_block_atom(molecule, node, force_field):
    """
    Check if the atom `node` of `molecule` is described by the block of
    its residue. Atoms that are not, are described by a modification.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
    node: abc.hashable
    force_field: :class:`vermouth.forcefield.ForceField`

    Returns
    -------
    bool
    """
    block = force_field.blocks.get(molecule.nodes[node]['resname'], None)
    return block is None or molecule.nodes[node]['atomname'] in block

def _annotate_modification(link, anchors, mod_name, seen_patterns):
    """
    Annotate the `anchors` of `link` with the name of the modification
    that has to be applied wherever the link matches. The annotation is
    a replace statement, so applying the link sets it on the molecule
    where the `ApplyModifications` processor picks it up.

    Two links that have the same matching pattern but annotate a
    different modification cannot be told apart when they are applied,
    which is reported.

    Parameters
    ----------
    link: :class:`vermouth.molecule.Link`
    anchors: abc.iterable
        the link atoms to annotate
    mod_name: str
        name of the modification
    seen_patterns: dict
        the modification name by matching pattern of the links that were
        annotated before; updated in place
    """
    for anchor in anchors:
        replace = link.nodes[anchor].get('replace', {})
        replace['annotated_modifications'] = [mod_name]
        link.nodes[anchor]['replace'] = replace

    # the pattern is what decides where a link matches; the non-edges
    # are part of it, because they are what distinguishes one terminus
    # from the other
    pattern = (tuple(sorted((node, link.nodes[node].get('resname')) for node in link.nodes)),
               tuple(sorted(tuple(sorted(edge)) for edge in link.edges)),
               tuple(sorted((anchor, attrs.get('atomname'), attrs.get('order'))
                            for anchor, attrs in link.non_edges)))
    if seen_patterns.get(pattern, mod_name) != mod_name:
        msg = ("The modifications {} and {} describe two residues that cannot be "
               "distinguished from the sequence alone. Give those residues "
               "different names or apply the modification by hand.")
        LOGGER.warning(msg, seen_patterns[pattern], mod_name)
    seen_patterns[pattern] = mod_name

def find_termini_mods(meta_molecule, molecule, force_field, modification_names=None):
    """
    Terminii are a bit special in the sense that they are often
    different from a repeat unit of the polymer in the polymer.

    If `modification_names` is given, the terminal residues that are
    described by one of those modifications are annotated with its name,
    such that the modification is applied whenever the generated link
    matches.

    Parameters
    ----------
    meta_molecule: :class:`networkx.Graph`
        residue graph of the molecule
    molecule: :class:`vermouth.molecule.Molecule`
    force_field: :class:`vermouth.forcefield.ForceField`
        the force-field the links are added to
    modification_names: dict[str, str]
        the name of a modification by the hash of the residue it
        describes as generated by `handle_ptms`
    """
    modification_names = modification_names or {}
    seen_patterns = {}
    terminal_nodes = [ node for node in meta_molecule.nodes if meta_molecule.degree(node) == 1 ]
    for meta_node in terminal_nodes:
        # get the node that is next to the terminal; by definition
        # it can only be one neighbor
        neigh_node = next(nx.neighbors(meta_molecule, meta_node))

        # some useful info
        neigh_resname = meta_molecule.nodes[neigh_node]['resname']
        resids = [meta_molecule.nodes[neigh_node]['resid'],
                  meta_molecule.nodes[meta_node]['resid']]
        ref_block = force_field.blocks[neigh_resname]
        target_block = meta_molecule.nodes[neigh_node]['graph']

        # find different properties
        replace_dict = defaultdict(dict)
        for node in target_block.nodes:
            target_attrs = target_block.nodes[node]
            # the residue can have more atoms than the block, because the
            # block is generated from the smallest version of a residue;
            # those extra atoms are described by a modification already
            # so they are skipped here
            if target_attrs['atomname'] not in ref_block:
                continue
            ref_attrs = ref_block.nodes[target_attrs['atomname']]
            for attr in ['atype', 'mass']:
                if target_attrs[attr] != ref_attrs[attr]:
                    replace_dict[node][attr] = target_attrs[attr]

        # bonded interactions could be different too so we need to check them;
        # this includes interactions entirely within the neighbor residue,
        # ones entirely within the terminal residue itself, and the bond(s)
        # that directly connect the two residues
        junction_atoms = list(target_block.nodes) + list(meta_molecule.nodes[meta_node]['graph'].nodes)
        # atoms that are not part of the block of their residue are described
        # by a modification. They cannot be part of the link, because a link
        # is applied before the modifications, when those atoms do not exist
        # yet. Their interactions are recorded by the modification instead
        junction_atoms = [node for node in junction_atoms
                          if _is_block_atom(molecule, node, force_field)]
        junction_block = molecule.subgraph(junction_atoms)
        overwrite_inters = defaultdict(list)
        for inter_type, inters in junction_block.interactions.items():
            versions = {}
            for target_inter in inters:
                mol_atoms_to_link_atoms, edges, resnames = _extract_edges_from_shortest_path(target_inter.atoms,
                                                                                             molecule,
                                                                                             min(resids))
                link_atoms =  [mol_atoms_to_link_atoms[atom] for atom in target_inter.atoms]
                # some of these interactions (e.g. the bond/angles/pairs that
                # span the junction itself) may already be specified, with
                # the same atoms and parameters, by a generic link that
                # extract_links produced earlier for this same pattern
                # elsewhere in the molecule (that link is not specific to
                # termini, so it also matches here); re-adding them in this
                # termini-specific link would just duplicate them once both
                # links are applied, so skip them here
                candidate = Interaction(atoms=link_atoms, parameters=target_inter.parameters, meta={})
                if _interaction_already_specified(candidate, inter_type, force_field.links):
                    continue
                if tuple(link_atoms) in versions:
                    n = versions[tuple(link_atoms)] + 1
                    meta = {"version": n}
                    versions[tuple(link_atoms)] = n
                else:
                    versions[tuple(link_atoms)] = 1
                    meta = {}
                link_inter = Interaction(atoms=link_atoms,
                                         parameters=target_inter.parameters,
                                         meta=meta)
                overwrite_inters[inter_type].append(link_inter)

        # we make a link; it spans the complete junction, because the
        # interactions collected above can involve any atom of the two
        # residues and the atoms have to be part of the link
        mol_atoms = junction_atoms
        link = vermouth.molecule.Link()
        mol_to_link, edges, resnames = _extract_edges_from_shortest_path(mol_atoms,
                                                                         molecule,
                                                                         min(resids))
        link_atoms = mol_to_link.values()
        link = vermouth.molecule.Link()
        link.add_nodes_from(link_atoms)
        for node in mol_atoms:
            link.nodes[mol_to_link[node]]['resname'] = molecule.nodes[node]['resname']
            if replace_dict[node]:
                link.nodes[mol_to_link[node]]['replace'] = replace_dict[node]

        force_field.links.append(link)
        for inter_type, inters in overwrite_inters.items():
            link.interactions[inter_type] += inters

        edges = find_connecting_edges(meta_molecule, molecule, [meta_node, neigh_node])
        for ndx, jdx in edges:
            link.add_edge(mol_to_link[ndx], mol_to_link[jdx])

        # without further constraints this link would also match at any
        # interior residue that happens to look like the terminal one
        # locally, because subgraph isomorphism does not care about extra
        # edges beyond those required by the link. A non-edge on the atom
        # that connects to the neighbor rules those out, by requiring that
        # it has no further neighbor on the side facing away from the
        # neighbor, i.e. that it is a genuine chain end
        outward_order = 1 if resids[1] > resids[0] else -1
        for ndx, jdx in edges:
            anchor = mol_to_link[ndx]
            non_edge_attrs = {'atomname': molecule.nodes[ndx]['atomname'],
                              'order': outward_order}
            link.non_edges.append([anchor, non_edge_attrs])

        # the terminal residue can be a version of its block that is
        # described by a modification. Which version it is cannot be told
        # from the sequence, so the link, which only matches at this very
        # terminus, annotates the atoms that connect to the neighbor with
        # the name of the modification
        if modification_names:
            ghash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
                        meta_molecule.nodes[meta_node]['graph'], node_attr='element')
            if ghash in modification_names:
                _annotate_modification(link,
                                       [mol_to_link[ndx] for ndx, _ in edges],
                                       modification_names[ghash],
                                       seen_patterns)

    return force_field

#: node attributes that are transferred from the residue graph
#: to the atoms of a modification
MOD_ATOM_ATTRS = ('atomname', 'element', 'atype', 'charge', 'mass')

#: node attributes of an atom that is already described by the block;
#: they are the criteria the modification is matched with
MOD_MATCH_ATTRS = ('atomname', 'element')

#: node attributes that only reflect where a residue sits within the
#: molecule; they differ between any two residues and thus are never
#: part of a modification
POSITION_ATTRS = ('index', 'resid', 'degree', 'charge_group')

def _attribute_diff(node_attrs, ref_attrs, tol=10**-6):
    """
    Collect those attributes of `node_attrs` that differ from the
    corresponding attribute in `ref_attrs`. Attributes that merely
    describe the position of the residue in the molecule are skipped
    (see `POSITION_ATTRS`) and floats are compared using `tol`,
    because charges are the result of an optimization.

    Parameters
    ----------
    node_attrs: dict
    ref_attrs: dict
    tol: float
        tolerance used when comparing floats

    Returns
    -------
    dict
        the differing attributes with the value of `node_attrs`
    """
    diff = {}
    for attr, value in node_attrs.items():
        if attr in POSITION_ATTRS:
            continue
        ref_value = ref_attrs.get(attr)
        if isinstance(value, float) and isinstance(ref_value, float):
            if not np.isclose(value, ref_value, rtol=0, atol=tol):
                diff[attr] = value
        elif value != ref_value:
            diff[attr] = value
    return diff

def _element_match(node1, node2):
    """
    Check if two node attribute dicts describe the same element.

    Parameters
    ----------
    node1: dict
    node2: dict

    Returns
    -------
    bool
    """
    return node1.get('element') == node2.get('element')

def _make_ptm_modification(graph, base_graph, graph_match, missing_atoms, name, tol=10**-6):
    """
    Generate a modification describing how the residue `graph` differs
    from the minimal residue `base_graph`.

    Following the vermouth specifications the `missing_atoms`, that is
    those atoms that are not part of the minimal residue, are the atoms
    only described by the modification and thus have the `PTM_atom`
    attribute set to True. All other atoms are already described by the
    block and have `PTM_atom` set to False. Of those atoms the
    modification records the anchors - i.e. the atoms of the minimal
    residue the missing atoms are bonded to - as well as all atoms that
    have attributes differing from the minimal residue. The differing
    attributes are stored in the `replace` attribute, such that they
    overwrite those of the block when the modification is applied. All
    edges between the recorded atoms are stored as well, such that the
    modification can be matched against a molecule.

    In addition all interactions that involve at least one of the
    missing atoms are recorded, because those are described neither by
    the block nor by any link. Any atom taking part in such an
    interaction becomes part of the modification. Interactions that
    only involve atoms of the minimal residue are not recorded, even if
    their parameters differ from those of the block.

    Note that vermouth expects a modification to be connected. If the
    missing and differing atoms describe more than one modification
    site of the same residue, the resulting modification is
    disconnected.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        graph of the residue that has the extra atoms; nodes must have
        the atomname and element attribute
    base_graph: :class:`networkx.Graph`
        graph of the minimal residue the block is generated from
    graph_match: dict
        mapping of the nodes of `graph` to those of `base_graph`
    missing_atoms: abc.iterable
        those nodes of `graph` that are not part of the minimal residue
    name: str
        name of the modification
    tol: float
        tolerance used when comparing float attributes

    Returns
    -------
    :class:`vermouth.molecule.Modification`
    """
    missing_atoms = set(missing_atoms)
    # the anchors are those atoms of the minimal residue to which the
    # missing atoms are attached
    anchors = set()
    for node in missing_atoms:
        anchors.update(set(nx.neighbors(graph, node)) - missing_atoms)

    modification = vermouth.molecule.Modification(name=name)
    # modifications, like blocks and links, are labelled by atomname
    mol_to_mod = {}

    # the atoms that are only described by the modification
    for node in missing_atoms:
        attrs = {attr: value for attr, value in graph.nodes[node].items()
                 if attr in MOD_ATOM_ATTRS}
        attrs['PTM_atom'] = True
        mol_to_mod[node] = attrs['atomname']
        modification.add_node(attrs['atomname'], **attrs)

    def _add_block_atom(node, replace=None):
        """
        Add the atom `node` of `graph`, which is already described by
        the block, to the modification. The modification is matched
        against the block, so the atom is labelled by the atomname of
        the minimal residue and the attributes of the minimal residue
        are the match criteria.
        """
        base_attrs = base_graph.nodes[graph_match[node]]
        attrs = {attr: value for attr, value in base_attrs.items()
                 if attr in MOD_MATCH_ATTRS}
        attrs['PTM_atom'] = False
        if replace:
            attrs['replace'] = replace
        mol_to_mod[node] = attrs['atomname']
        modification.add_node(attrs['atomname'], **attrs)

    # the atoms that are already described by the block; they are only
    # part of the modification if they anchor a missing atom or if any
    # of their attributes differs from the minimal residue
    for node, base_node in graph_match.items():
        replace = _attribute_diff(graph.nodes[node],
                                  base_graph.nodes[base_node],
                                  tol=tol)
        if node not in anchors and not replace:
            continue
        _add_block_atom(node, replace=replace)

    # all interactions that involve at least one of the missing atoms are
    # described neither by the block nor by any link, because the block
    # only has the interactions of the minimal residue and links only
    # cover interactions spanning more than one residue
    for inter_type, interactions in graph.interactions.items():
        versions = {}
        for interaction in interactions:
            if missing_atoms.isdisjoint(interaction.atoms):
                continue
            # an interaction can reach beyond the anchors, so it may
            # involve atoms that are not part of the modification yet
            for atom in interaction.atoms:
                if atom not in mol_to_mod:
                    _add_block_atom(atom)
            new_inter = _relabel_interaction_atoms(interaction, mol_to_mod)
            # multiple interactions of the same type between the same
            # atoms are distinguished by the version meta attribute
            count = versions.get(tuple(new_inter.atoms), 0) + 1
            versions[tuple(new_inter.atoms)] = count
            meta = dict(new_inter.meta)
            if count > 1:
                meta['version'] = count
            modification.interactions[inter_type].append(new_inter._replace(meta=meta))

    for node, neigh_node in graph.subgraph(mol_to_mod.keys()).edges:
        modification.add_edge(mol_to_mod[node], mol_to_mod[neigh_node])

    return modification

def find_minimal_residue(graph_group, hash_group):
    """
    Given a group of residue graphs that share the same resname, find
    the smallest of them and describe all others as modifications of
    that minimal residue.

    Every graph in `graph_group` has to contain the minimal residue as
    node induced subgraph, where nodes are matched by element. Those
    atoms that are not part of the subgraph match together with their
    anchors are recorded as a :class:`vermouth.molecule.Modification`.
    Modifications are named after the resname and the graph hash of
    the residue they belong to, in the same way blocks are.

    Parameters
    ----------
    graph_group: abc.iterable[:class:`networkx.Graph`]
        graphs of the residues sharing the same resname; nodes must
        have the atomname and element attribute
    hash_group: abc.iterable[str]
        the graph hash of each graph in `graph_group` in the same order

    Returns
    -------
    tuple(:class:`networkx.Graph`, dict[str, :class:`vermouth.molecule.Modification`])
        the graph of the minimal residue and the modifications
        describing all other residues by the hash of the residue they
        belong to; the name of a modification is stored with it

    Raises
    ------
    IOError
        if a residue does not contain the minimal residue as subgraph
    """
    base_graph = min(graph_group, key=len)
    modifications = {}
    for graph, ghash in zip(graph_group, hash_group):
        # the match maps the nodes of the residue to those of the
        # minimal residue; all atoms not taking part in the match are
        # extra atoms only described by the modification
        graph_match = find_one_subgraph_match(graph,
                                              base_graph,
                                              node_match=_element_match)
        resname = graph.nodes[next(iter(graph.nodes))].get('resname')
        if graph_match is None:
            msg = (f"Residue {resname} comes in different versions, but not all "
                    "of them contain the smallest version as subgraph. Thus no "
                    "modifications can be generated.")
            raise IOError(msg)

        missing_atoms = set(graph.nodes) - set(graph_match.keys())
        # this residue is the minimal residue itself
        if not missing_atoms:
            continue

        name = f"{resname}-{ghash}"
        modifications[ghash] = _make_ptm_modification(graph,
                                                     base_graph,
                                                     graph_match,
                                                     missing_atoms,
                                                     name)

    return base_graph, modifications

def handle_ptms(topology,
                unique_fragments,
                res_graph,
                target_mol,
                force_field,
                crg_dict):
    """
    Generate a block for every residue name and describe those residues
    that come in more than one version as modifications of the smallest
    version, which is the one the block is generated from.

    Parameters
    ----------
    topology: :class:`polyply.src.topology.Topology`
    unique_fragments: dict[str, :class:`networkx.Graph`]
        the residue graphs by their graph hash
    res_graph: :class:`networkx.Graph`
        residue graph of the target molecule
    target_mol: :class:`vermouth.molecule.Molecule`
    force_field: :class:`vermouth.forcefield.ForceField`
        the force-field the blocks and modifications are added to
    crg_dict: dict[str, float]
        the total charge by residue name

    Returns
    -------
    dict[str, str]
        the name of the modification by the hash of the residue it
        describes
    """
    # collect residue names and graph hashes
    hash_groups = defaultdict(list)
    graph_groups = defaultdict(list)
    for ghash, graph in unique_fragments.items():
        resname = graph.nodes[next(iter(graph.nodes))].get('resname')
        hash_groups[resname].append(ghash)
        balance_charges(graph,
                        topology=topology,
                        charge=float(crg_dict[resname]))
        graph_groups[resname].append(graph)

    modification_names = {}
    for resname in hash_groups:
        fragment = graph_groups[resname][0]
        # here we have to deal with a residue that does has the same
        # resname for multiple residues
        if len(set(hash_groups[resname])) != 1:
            fragment, modifications = find_minimal_residue(graph_groups[resname],
                                                           hash_groups[resname])
            for ghash, modification in modifications.items():
                force_field.modifications[modification.name] = modification
                modification_names[ghash] = modification.name

        new_block = extract_block(target_mol, fragment, defines={})
        nx.set_node_attributes(new_block, 1, "resid")
        new_block.nrexcl = target_mol.nrexcl
        force_field.blocks[resname] = new_block

    return modification_names
