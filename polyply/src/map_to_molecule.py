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

import networkx as nx
from vermouth.graph_utils import make_residue_graph
from polyply.src.processor import Processor

def tag_exclusions(node_to_block, force_field):
    """
    Given block names matching nodes in the meta_molecule
    graph check if the corresponding blocks in `force_field`
    all have the same number of default exclusions. If not
    find the minimum number of exclusions and tag all
    nodes with the original exclusion number. Then
    change the exclusion number to the lowest value.

    Note the tag is picked up by apply links where
    the excluions are generated.
    """
    excls = {}
    for node in node_to_block:
        block = force_field.blocks[node_to_block[node]]
        excls[node] = block.nrexcl

    if len(set(excls.values())) > 1:
        min_excl = min(list(excls.values()))
        for node, excl in excls.items():
            block = force_field.blocks[node_to_block[node]]
            nx.set_node_attributes(block, excl, "exclude")
            block.nrexcl = min_excl

def _correspondence_to_residue(meta_molecule,
                               molecule,
                               correspondence,
                               res_node):
    """
    Given a `meta_molecule` and the underlying higher resolution
    `molecule` as well as a correspondence dict, describing how
    a single node (res_node) in meta_molecule corresponds to a
    fragment in molecule make a graph of that residue and propagate
    all meta_molecule node attributes to that graph.

    Parameters
    ----------
    meta_molecule: polyply.src.meta_molecule.MetaMolecule
        The meta molecule to process.
    molecule: vermouth.molecule.Molecule
    correspondance: list
    res_node: abc.hashable
        the node in meta_molecule
    """
    resid = meta_molecule.nodes[res_node]["resid"]
    residue = nx.Graph()
    for mol_node in correspondence.values():
        data = molecule.nodes[mol_node]
        if data["resid"] == resid:
            residue.add_node(mol_node, **data)
            for attribute, value in meta_molecule.nodes[res_node].items():
                # graph and seqID are specific attributes set by make residue
                # graph or the gen_seq tool, which we don't want to propagate.
                if attribute in ["graph", "seqID"]:
                    continue
                residue.nodes[mol_node][attribute] = value

    return residue

MultiblockError = ("Block {} seems to represent more than a single residue,\n"
                   "but within the residue graph only a single corresponding node\n"
                   "was found. If you have blocks representing multiresidues\n"
                   "provide the full residue graph and label all nodes that\n"
                   "corresponding to multiresidues with the \"from_itp\" label.")

def _assert_blocks_in_FF(block_names, force_field):
    for name in block_names:
        if name not in force_field.blocks:
            raise IOError("Couldn't find block with residue name {name}\n"
                          "in the library or input file definitions.".format(name=name))

class MapToMolecule(Processor):
    """
    This processor takes a :class:`MetaMolecule` and generates a
    :class:`vermouth.molecule.Molecule`, which consists at this stage
    of disconnected blocks. These blocks can be connected using the
    :class:`ApplyLinks` processor.
    """
    def __init__(self, force_field):
        self.node_to_block = {}
        self.node_to_fragment = {}
        self.fragments = []
        self.multiblock_correspondence = []
        self.added_fragment_nodes = []
        self.force_field = force_field

    def match_nodes_to_blocks(self, meta_molecule):
        """
        This function matches the nodes in the meta_molecule
        to the blocks in the force-field. It does the essential
        bookkeeping for two cases and populates the node_to_block,
        and node_to_fragment dicts as well as the fragments attribute.
        It distinguishes two cases:

        1) the node corresponds to a single residue block; here
           node_to_block entry is simply the resname of the block
        2) the node has the from_itp attribute; in this case
           the node is part of a multiresidue block in the FF,
           all nodes corresponding to that block form a fragment.
           All fragments are added to the fragments attribute, and
           the nodes in those fragments all have the entry node_to_block
           set to the block. In addition it is recorded to which fragment
           specifically the node belongs in the node_to_fragment dict.

        Parameters
        ----------
        meta_molecule: polyply.src.meta_molecule.MetaMolecule
            The meta molecule to process.

        """
        regular_graph = nx.Graph()
        restart_graph = nx.Graph()
        restart_attr = nx.get_node_attributes(meta_molecule, "from_itp")
        restart_graph.add_nodes_from(restart_attr.keys())

        # in case we only have a single residue
        # we assume the user is sane and that residue is not from
        # an itp file
        if len(meta_molecule.nodes) == 1:
            regular_graph.add_nodes_from(meta_molecule.nodes)

        # this will falsely also connect two molecules with the
        # same molecule name, if they are from_itp and consecutively
        # we deal with that below
        for idx, jdx in nx.dfs_edges(meta_molecule):
            if idx in restart_attr and jdx in restart_attr:
                if restart_attr[idx] == restart_attr[jdx]:
                    restart_graph.add_edge(idx, jdx)
            else:
                regular_graph.add_edge(idx, jdx)

        # regular nodes have to match a block in the force-field by resname
        for node in regular_graph.nodes:
            self.node_to_block[node] = meta_molecule.nodes[node]["resname"]

        # fragment nodes match parts of blocks, which describe molecules
        # with more than one residue. Sometimes multiple molecules come
        # after each other in that case the connected component needs to
        # be an integer multiple of the block
        n_fragments = 0
        for fragment in nx.connected_components(restart_graph):
            frag_nodes = list(fragment)
            block = self.force_field.blocks[restart_attr[frag_nodes[0]]]
            block_res = make_residue_graph(block, attrs=('resid', 'resname'))
            len_block = len(block_res)
            len_frag = len(fragment)
            # here we check if the fragment is an integer multiple
            # of the multiresidue block
            if len_frag%len_block == 0:
                n_blocks = len_frag//len_block
            else:
                # if it is not raise an error
                molname = restart_attr[frag_nodes[0]]
                msg = (f"When mapping the molecule {molname} onto the residue graph "
                        "nodes labeled with from_itp, a mismatch in the length between "
                        "the provided molecule and the residue graph is found. Make "
                        "sure that all residues are in the residue graph and input itp-file.")
                raise IOError(msg)

            for fdx in range(0, n_blocks):
                current_frag_nodes = frag_nodes[fdx*len_block: (fdx+1)*len_block]
                self.fragments.append(current_frag_nodes)
                for node in current_frag_nodes:
                    self.node_to_block[node] = restart_attr[frag_nodes[0]]
                    self.node_to_fragment[node] = n_fragments
                n_fragments += 1

    def add_blocks(self, meta_molecule):
        """
        Add disconnected blocks to :class:`vermouth.molecule.Moleclue`
        and set the graph attribute to meta_molecule matching the node
        with the underlying fragment it represents at higher resolution.
        Note that this function also takes care to properly add multi-
        residue blocks (i.e. from an existing itp-file).

        Parameters
        ----------
        meta_molecule: polyply.src.meta_molecule.MetaMolecule
            The meta molecule to process.

        Returns
        -------
        vermouth.molecule.Molecule
            The disconnected fine-grained molecule.
        """
        # get a defined order for looping over the resiude graph
        node_keys = list(meta_molecule.nodes())
        resid_dict = nx.get_node_attributes(meta_molecule, "resid")
        resids = [resid_dict[node] for node in node_keys]
        node_keys = [x for _, x in sorted(zip(resids, node_keys))]
        # get the first node and convert it to molecule
        start_node = node_keys[0]
        new_mol = self.force_field.blocks[self.node_to_block[start_node]].to_molecule()

        # in this case the node belongs to a fragment for which there is a
        # multiresidue block
        if "from_itp" in meta_molecule.nodes[start_node]:
            # add all nodes of that fragment to added_fragment nodes
            fragment_nodes = list(self.fragments[self.node_to_fragment[start_node]])
            self.added_fragment_nodes += fragment_nodes

            # extract the nodes of this paticular residue and store a
            # dummy correspndance
            correspondence = {node:node for node in new_mol.nodes}
            self.multiblock_correspondence.append({node:node for node in new_mol.nodes})
            residue = _correspondence_to_residue(meta_molecule,
                                                 new_mol,
                                                 correspondence,
                                                 start_node)
            # add residue to meta_molecule node
            meta_molecule.nodes[start_node]["graph"] = residue
        else:
            # if the block represents more than one residue all residues
            # have to be already in the meta_molecule and they have to be
            # labelled as from_itp. If not we raise an error
            if len(set(nx.get_node_attributes(new_mol, "resid").values())) > 1:
                raise IOError(MultiblockError.format(meta_molecule.nodes[start_node]["resname"]))

            # we store the block together with the residue node
            meta_molecule.nodes[start_node]["graph"] = new_mol.copy()

        # now we loop over the rest of the nodes
        for node in node_keys[1:]:
            # in this case the node belongs to a fragment which has been added
            # we only extract the residue belonging to this paticular node
            if node in self.added_fragment_nodes:
                fragment_id = self.node_to_fragment[node]
                correspondence = self.multiblock_correspondence[fragment_id]
            # in this case we have to add the node from the block definitions
            else:
                block = self.force_field.blocks[self.node_to_block[node]]
                # if the block represents more than one residue all residues
                # have to be already in the meta_molecule and they have to be
                # labelled as from_itp. If not we raise an error
                if "from_itp" not in meta_molecule.nodes[node] and \
                len(set(nx.get_node_attributes(block, "resid").values())) > 1:

                    raise IOError(MultiblockError.format(self.node_to_block[node]))

                correspondence = new_mol.merge_molecule(block)

            # make the residue from the correspondence
            residue = _correspondence_to_residue(meta_molecule,
                                                 new_mol,
                                                 correspondence,
                                                 node)
            # add residue to node
            meta_molecule.nodes[node]["graph"] = residue

            # in this case we just added a new multiblock residue so we store
            # the correspondence as well as keep track of the nodes that are
            # part of that fragment
            if "from_itp" in meta_molecule.nodes[node] and node not in self.added_fragment_nodes:
                fragment_nodes = list(self.fragments[self.node_to_fragment[node]])
                self.added_fragment_nodes += fragment_nodes
                self.multiblock_correspondence.append(correspondence)

        return new_mol

    def run_molecule(self, meta_molecule):
        """
        Take a meta_molecule and generated a disconnected graph
        of the higher resolution molecule by matching the resname
        attribute to blocks in the force-field. This function
        also takes care to correcly add parameters from an itp
        file to fine-grained molecule. It also sets the 'graph'
        attribute, which is the higher-resolution fragment that
        the meta_molecule node represents.

        Parameters
        ----------
        molecule: polyply.src.meta_molecule.MetaMolecule
             The meta molecule to process.

        Returns
        -------
        molecule: polyply.src.meta_molecule.MetaMolecule
            The meta molecule with attribute molecule that is the
            fine grained molecule.
        """
        # first we match the residue names to blocks
        # in the force-field. Residue names can also be part
        # of a larger fragment stored as a block or refer to
        # a block which consists of multiple residues. This
        # gets entangled here
        self.match_nodes_to_blocks(meta_molecule)
        # raise an error if a block is not known to the library
        _assert_blocks_in_FF(self.node_to_block.values(),
                             self.force_field)
        # next we check if all exclusions are the same and if
        # not we adjust it such that the lowest exclusion number
        # is used. ApplyLinks then generates those appropiately
        tag_exclusions(self.node_to_block, self.force_field)
        # now we add the blocks generating a new molecule
        new_molecule = self.add_blocks(meta_molecule)
        meta_molecule.molecule = new_molecule
        return meta_molecule
