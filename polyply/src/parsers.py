from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import vermouth
import vermouth.gmx
from vermouth.gmx.itp_read import ITPDirector

class PolyplyParser(ITPDirector):
    '''
    Parser for polyply input format.
    '''

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
        new_atoms = []

        # the uncommented statement does not work because node and
        # atom name are couple for blocks, which is debatably useful
        #atom_names = list(nx.get_node_attributes(block, 'atomname'))
        atom_names = [block.nodes[node]["atomname"] for node in block.nodes]

        # at this stage the link only has a single interaction
        interaction = link.interactions[inter_type][-1]
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
        return link

    def _split_links_and_blocks(self, block):

        # Make sure to add the atomtype resdidue number etc to
        # the proper nodes.

        n_atoms = len(block.nodes)
        res_name = block.nodes[0]['resname']
        for key in block.interactions:
            block_interactions = []
            for interaction in block.interactions[key]:
                new_link = vermouth.molecule.Link()
                new_link.interactions = defaultdict(list)
                new_link.name = res_name
                if np.sum(np.array(interaction.atoms) > n_atoms - 1) > 0:
                    new_link.interactions[key].append(interaction)
                    self._treat_link_atoms(block, new_link, key)
                    new_link.make_edges_from_interaction_type(type_=key)
                    self.force_field.links.append(new_link)
                else:
                    block_interactions.append(interaction)

            block.interactions[key] = block_interactions


    # overwrites the finalize method to deal with dangling bonds
    # and to deal with multiple interactions in the way needed
    # for polyply to work
    def finalize(self, lineno=0):

        if self.current_meta is not None:
            raise IOError("Your #ifdef section is orderd incorrectly."
                          "There is no #endif for the last pragma..")

        prev_section = self.section
        self.section = []
        self.finalize_section(prev_section, prev_section)
        self.macros = {}
        self.section = None

        for block in self.force_field.blocks.values():
            if len(block.nodes) > 0:
                n_atoms = len(block.nodes)
                self._split_links_and_blocks(block)
                self.treat_link_multiple()

def read_polyply(lines, force_field):
    director = PolyplyParser(force_field)
    return list(director.parse(iter(lines)))
