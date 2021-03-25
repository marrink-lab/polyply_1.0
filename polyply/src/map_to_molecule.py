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
from polyply.src.processor import Processor

def tag_exclusions(node_to_block, force_field):
    """
    Given the names of some `blocks` check if the
    corresponding molecules in `force_field` have
    the same number of default exclusions. If not
    find the minimum number of exclusions and tag all
    nodes with the original exclusion number. Then
    change the exclusion number.

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

class MapToMolecule(Processor):
    """
    This processor takes a :class:`MetaMolecule` and generates a
    :class:`vermouth.molecule.Molecule`, which consists at this stage
    of disconnected blocks. These blocks can be connected using the
    :class:`ConnectMolecule` processor. It can either run on a
    single meta molecule or a system. The later is currently not
    implemented.
    """
    def __init__(self, force_field):
        self.node_to_block = {}
        self.node_to_fragment = {}
        self.fragments = []
        self.multiblock_correspondence = []
        self.added_fragments = []
        self.added_fragment_nodes = []
        self.force_field = force_field

    def match_nodes_to_blocks(self, meta_molecule):
        """
        For all nodes in the meta_molecule match them either to a
        block in the force_field or to a block that is a multi
        residue molecule, in which case the node has the restart
        attribute. There is a third option in which a resname matches
        a block, which is itself a multiresidue block.
        """
        regular_graph = nx.Graph()
        restart_graph = nx.Graph()
        restart_attr = nx.get_node_attributes(meta_molecule, "from_itp")

        # this breaks down when to proteins are directly linked
        # but that is an edge-case we can worry about later
        for idx, jdx in nx.dfs_edges(meta_molecule):
            # the two nodes are restart nodes
            if idx in restart_attr and jdx in restart_attr:
                restart_graph.add_edge(idx, jdx)
            else:
                regular_graph.add_edge(idx, jdx)

        # regular nodes have to mactch a block in the force-field by resname
        for node in regular_graph.nodes:
            self.node_to_block[node] = meta_molecule.nodes[node]["resname"]

        # restart nodes are match a block in the force-field but are themselves
        # only part of that block
        for fragment in nx.connected_components(restart_graph):
            block_name = restart_attr[list(fragment)[0]]
            if all([restart_attr[node] == block_name  for node in fragment]):
                self.fragments.append(fragment)
                block = self.force_field.blocks[block_name]
                for node in fragment:
                    self.node_to_block[node] = block_name
                    self.node_to_fragment[node] = len(self.fragments) - 1
            else:
                raise IOError

    def add_blocks(self, meta_molecule):
        """
        Add disconnected blocks to :class:`vermouth.molecule.Molecue`
        and if a multiresidue block is encountered expand the meta
        molecule graph to include the block at residue level.
        """
        # get a defined order for looping over the resiude graph
        node_keys = list(meta_molecule.nodes())
        resid_dict = nx.get_node_attributes(meta_molecule, "resid")
        resids = [ resid_dict[node] for node in node_keys]
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
            if "from_itp" in meta_molecule.nodes[node] and node not in self.added_fragments:
                fragment_nodes = list(self.fragments[self.node_to_fragment[start_node]])
                self.added_fragment_nodes += fragment_nodes
                self.multiblock_correspondence.append(correspondence)


        return new_mol

    def run_molecule(self, meta_molecule):
        """
        Process a single molecule. Must be implemented by subclasses.
        Parameters
        ----------
        molecule: polyply.src.meta_molecule.MetaMolecule
             The meta molecule to process.
        Returns
        -------
        vermouth.molecule.Molecule
            Either the provided molecule, or a brand new one.
        """
        # in a first step we match the residue names to blocks
        # in the force-field. Residue names can also be part
        # of a larger fragment stored as a block or refer to
        # a block which consists of multiple residues. This
        # gets entangled here
        self.match_nodes_to_blocks(meta_molecule)
        # next we check if all exclusions are the same and if
        # not we adjust it such that the lowest exclusion number
        # is used. ApplyLinks then generates those appropiately
        tag_exclusions(self.node_to_block, self.force_field)
        # now we add the blocks generating a new molecule
        new_molecule = self.add_blocks(meta_molecule)
        meta_molecule.molecule = new_molecule
        return meta_molecule
