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
import collections
from vermouth.parser_utils import (
    SectionLineParser, _tokenize, _substitute_macros, _parse_macro
)
from vermouth.ffinput import FFDirector, _get_atoms, _treat_atom_prefix, _parse_edges

class PolyplyFFParser(FFDirector):
    '''
    More strictly implements differences between .ff within polyply
    and .ff within martinize2.
    '''
    @SectionLineParser.section_parser('moleculetype', 'edges',
                                      negate=False, context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'non-edges',
                                      negate=True, context_type='block')
    @SectionLineParser.section_parser('link', 'edges',
                                      negate=False, context_type='link')
    @SectionLineParser.section_parser('modification', 'edges',
                                      negate=False, context_type='modification')

    def _edges(self, line, lineno=0, negate=False, context_type=''):
        context = self.get_context(context_type)
        tokens = collections.deque(_tokenize(line))
        _parse_edges_new(tokens, context, context_type, negate=negate)


    @SectionLineParser.section_parser('link', 'non-edges',
                                      negate=True, context_type='link')
    def _non_edges(self, line, lineno, negate=False, context_type=''):
        context = self.get_context(context_type)
        tokens = collections.deque(_tokenize(line))
        _parse_edges(tokens, context, context_type, negate)

def _parse_edges_new(tokens, context, context_type, negate):
    """
    Parse the edge directive and add edge-attributes when
    required.
    """
    if negate:
        raise IOError('The "non-edges" section is only valid in links.')

    atoms = _get_atoms(tokens, natoms=2)
    prefixed_atoms = []
    for idx, atom in enumerate(atoms):
        prefixed_reference, attributes = _treat_atom_prefix(*atom)
        for key in ["atomname", "order", "resname"]:
            attributes.pop(key, None)
        prefixed_atoms.append(prefixed_reference)

        if idx == 0 and attributes:
            raise IOError("The edge directive does not allow to set node attributes.")
        else:
            edge_attributes = attributes

    error_message = 'Atom with name {} not found for {} {}'
    for prefixed_atom in prefixed_atoms:
        atomname = prefixed_atom[0]
        if atomname not in context and context_type == 'modification':
            raise KeyError(error_message.format(atomname, context_type,
                                                context.name))
    context.add_edge(prefixed_atoms[0], prefixed_atoms[1], **edge_attributes)

def read_ff(lines, force_field):
    director = PolyplyFFParser(force_field)
    return list(director.parse(iter(lines)))
