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

from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import vermouth
import vermouth.gmx
from vermouth.parser_utils import SectionLineParser
from vermouth.gmx.itp_read import ITPDirector

class PolyplyParser(ITPDirector):
    '''
    Parser for polyply input format.
    '''
    def __init__(self, force_field):
        super().__init__(force_field)
        self.citations = set()

    @SectionLineParser.section_parser('moleculetype', 'citation')
    def _parse_citation(self, line, lineno=0):
        cite_keys = line.split()
        self.current_block.citations.update(cite_keys)

    @SectionLineParser.section_parser('citations')
    def _pase_ff_citations(self, line, lineno=0):
        # parses force-field wide citations
        cite_keys = line.split()
        self.citations.update(cite_keys)

    # overwritten to allow for dangling bonds
    def _treat_block_interaction_atoms(self, atoms, context, section):
        all_references = []
        for atom in atoms:
            reference = atom[0]
            if reference.isdigit():
                if int(reference) < 1:
                    msg = ('In section {} is a negative atom reference, which is not allowed.')
                    raise IOError(msg.format(section.name))

               # The indices in the file are 1-based
                reference = int(reference) - 1
                atom[0] = reference
            else:
                msg = ('Atom names in blocks cannot be prefixed with + or -. '
                       'The name "{}", used in section "{}" of the block "{}" '
                       'is not valid in a block.')
                raise IOError(msg.format(reference, section, context.name))
            all_references.append(reference)
        return all_references

    def treat_link_multiple(self):
        """
        Iterates over a :class:`vermouth.force_field.ForceField` and
        adds version tags for all interactions within a
        :class:`vermouth.molecule.Link` that are applied to the same atoms.
        """
        for link in self.force_field.links:
            for key in link.interactions:
                terms = link.interactions[key]
                count_terms = Counter(tuple(term.atoms) for term in terms)
                for term in terms:
                    tag = count_terms[tuple(term.atoms)]
                    if tag >= 1:
                        term.meta.update({"version":tag})
                        count_terms[tuple(term.atoms)] = tag -1

    def _treat_link_atoms(self, block, link, inter_type):

        # we need to convert the atom index to an atom-name
        n_atoms = len(block.nodes)
        # the uncommented statement does not work because node and
        # atom name are couple for blocks, which is debatably useful
        #atom_names = list(nx.get_node_attributes(block, 'atomname'))
        atom_names = [block.nodes[node]["atomname"] for node in block.nodes]

        for inter_type in link.interactions:
            for interaction in link.interactions[inter_type]:
                new_atoms = []
                for atom in interaction.atoms:
                    prefix = ""
                    while atom/n_atoms >= 1:
                        atom = atom - n_atoms
                        prefix = prefix + "+"

                    new_name = prefix + atom_names[atom]
                    new_atoms.append(new_name)
                    attrs = block.nodes[atom]
                    link.add_node(new_name, **attrs)
                    order = prefix.count("+")
                    nx.set_node_attributes(link, {new_name:order}, "order")

                interaction.atoms[:] = new_atoms

        return new_atoms

    def _split_links_and_blocks(self, block):

        # Make sure to add the atomtype resdidue number etc to
        # the proper nodes.

        n_atoms = len(block.nodes)
        res_name = block.name
        prev_atoms = []
        links = []
        for key in block.interactions:
            block_interactions = []
            for interaction in block.interactions[key]:
                if any(isinstance(atom, str) for atom in interaction.atoms):
                   return

                if np.sum(np.array(interaction.atoms) > n_atoms - 1) > 0:
                   if interaction.atoms != prev_atoms:
                       prev_atoms[:] = interaction.atoms
                       new_link = vermouth.molecule.Link()
                       new_link.interactions = defaultdict(list)
                       new_link.citations = block.citations
                       new_link.name = res_name
                       links.append(new_link)
                   links[-1].interactions[key].append(interaction)
                else:
                    block_interactions.append(interaction)

            block.interactions[key] = block_interactions

        for link in links:
            self._treat_link_atoms(block, link, key)
            self.force_field.links.append(link)

    def _make_edges(self):
       for block in self.force_field.blocks.values():
           inter_types = list(block.interactions.keys())
           for inter_type in inter_types:
               block.make_edges_from_interaction_type(type_=inter_type)

       for link in self.force_field.links:
           inter_types = list(link.interactions.keys())
           for inter_type in inter_types:
               link.make_edges_from_interaction_type(type_=inter_type)

    # overwrites the finalize method to deal with dangling bonds
    # and to deal with multiple interactions in the way needed
    # for polyply to work

    def finalize(self, lineno=0):

        if self.current_meta is not None:
            raise IOError("Your #ifdef/#ifndef section is orderd incorrectly."
                          "There is no #endif for the last pragma..")

        prev_section = self.section
        self.section = []
        self.finalize_section(prev_section, prev_section)
        self.macros = {}
        self.section = None

        for block in self.force_field.blocks.values():
            block.citations.update(self.citations)
            if len(block.nodes) > 0:
                n_atoms = len(block.nodes)
                self._split_links_and_blocks(block)
                self.treat_link_multiple()
        self._make_edges()

def read_polyply(lines, force_field):
    director = PolyplyParser(force_field)
    return list(director.parse(iter(lines)))
