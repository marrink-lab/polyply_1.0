# Copyright 2018 University of Groningen
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

"""
Read GROMACS .itp files.

"""

import collections
import copy
import numbers
import json
from vermouth.molecule import (
    Block, Link,
    Interaction, DeleteInteraction,
    Choice, NotDefinedOrNot,
    ParamDistance, ParamAngle, ParamDihedral, ParamDihedralPhase,
)
from vermouth.parser_utils import (
    SectionLineParser, _tokenize, _parse_macro, _substitute_macros,
)

# Python 3.4 does not raise JSONDecodeError but ValueError.

try:
    from json import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

VALUE_PREDICATES = {
    'not': NotDefinedOrNot,
}

PARAMETER_EFFECTORS = {
    'dist': ParamDistance,
    'angle': ParamAngle,
    'dihedral': ParamDihedral,
    'dihphase': ParamDihedralPhase,
}


class ITPDirector(SectionLineParser):
    COMMENT_CHAR = ';'
    interactions_natoms = {
        'bonds': 2,
        'angles': 3,
        'dihedrals': 4,
        'impropers': 4,
        'constraints': 2,
        'virtual_sites2': 3,
        'virtual_sites3': 4,
        'virtual_sites4': 5,
        'pairs': 2,
        'pairs_nb':2,
        'position_restraints':1,
        'distance_restraints':2,
        'dihedral_restraints':4,
        'orientation_restraints':2,
        'angle_restraints':4,
        'angle_restraints_z':2
    }

    def __init__(self, force_field):

        super().__init__()
        self.force_field = force_field
        self.current_block = None
        self.current_meta = None
        self.blocks = collections.OrderedDict()

        self.header_actions = {
            ('moleculetype', ): self._new_block
        }

    def dispatch(self, line):
        """
        Looks at `line` to see what kind of line it is, and returns either
        :meth:`parse_header` if `line` is a section header or
        :meth:`parse_section` otherwise. Calls :meth:`is_section_header` to see
        whether `line` is a section header or not.

        Parameters
        ----------
        line: str

        Returns
        -------
        collections.abc.Callable
            The method that should be used to parse `line`.
        """

        if self.is_section_header(line):
            return self.parse_header
        elif self.is_def(line):
            return self.parse_def
        else:
            return self.parse_section

    def is_def(self, line):
        """
        Parameters
        ----------
        line: str
            A line of text.

        Returns
        -------
        bool
            ``True`` iff `line` is a def statement.

        """
        if line.startswith('#') :
           return True
        else:
           return False


    def parse_def(self, line, lineno=0):
        """
        Parses the beginning and end of define sections
        with line number `lineno`. Sets :attr:`current_meta`
        when applicable. Does check if ifdefs overlap.

        Parameters
        ----------
        line: str
        lineno: str

        Returns
        -------
        object
            The result of calling :meth:`finalize_section`, which is called
            if a section ends.

        Raises
        ------
        IOError
            If the def sections are missformatted
        """
                        
        if line == '#endif' and self.current_meta != None:
           self.current_meta=None

        elif line == '#endif' and self.current_meta == None:
           raise IOError("Your #ifdef section is orderd incorrectly." 
			 "At line {} I read #endif but I haven not read"
                         "a ifdef before.".format(lineno)) from error


        elif line != '#endif' and self.current_meta == None:
           condition, tag = line.split()
           self.current_meta = {'tag':tag,'condition':condition}

        else:
           raise IOError("Your #ifdef section is orderd incorrectly." 
			 "At line {} I read #endif but I haven not read"
                         "a ifdef before.".format(lineno)) from error

    def parse_header(self, line, lineno=0):
        """
        Parses a section header with line number `lineno`. Sets :attr:`section`
        when applicable. Does not check whether `line` is a valid section
        header.

        Parameters
        ----------
        line: str
        lineno: str

        Returns
        -------
        object
            The result of calling :meth:`finalize_section`, which is called
            if a section ends.

        Raises
        ------
        KeyError
            If the section header is unknown.
        """
                        
        prev_section = self.section

        ended = []
        section = self.section + [line.strip('[ ]').casefold()]
        if tuple(section[-1:]) in self.METH_DICT:
            self.section = section[-1:]
        else:
            while (tuple(section) not in self.METH_DICT
                   and len(section) > 1):
                ended.append(section.pop(-2))  # [a, b, c, d] -> [a, b, d]
            self.section = section
                          
        result = None

        if len(prev_section) != 0:
            print(len(prev_section))
            result = self.finalize_section(prev_section, ended)

        action = self.header_actions.get(tuple(self.section))
        if action:
            action()
                           
        return result

    def finalize_section(self, pervious_section, ended_section):

        """
        Called once a section is finished. It appends the current_links list
        to the links and update the block dictionary with current_block. Thereby it
        finishes the reading a given section. 

        Parameters
        ---------
        previous_section: list[str]
            The last parsed section.
        ended_section: list[str]
            The sections that have been ended.
        """
        
        if self.current_block is not None:
           self.force_field.blocks[self.current_block.name] = self.current_block
       
    def _new_block(self):
        self.current_block = Block(force_field=self.force_field)

    @SectionLineParser.section_parser('moleculetype')
    def _block(self, line, lineno=0):
        name, nrexcl = line.split()
        self.current_block.name = name
        self.current_block.nrexcl = int(nrexcl)

    @SectionLineParser.section_parser('moleculetype', 'atoms')
    def _block_atoms(self, line, lineno=0):
        tokens = collections.deque(_tokenize(line))
        _parse_block_atom(tokens, self.current_block)

    @SectionLineParser.section_parser('moleculetype', 'bonds', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'angles', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'dihedrals', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'impropers', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'constraints', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'pairs', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'exclusions', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites2', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'position_restraints', context_type='block')

    def _interactions(self, line, lineno=0, context_type=''):

        context = self.current_block
        interaction_name = self.section[-1]
        delete = False

        tokens = collections.deque(_tokenize(line))

        n_atoms = self.interactions_natoms.get(interaction_name)

        _base_parser(
            tokens,
            context,
            context_type=context_type,
            section=interaction_name,
            natoms=n_atoms,
            delete=delete,
            current_meta=self.current_meta
            )


    @SectionLineParser.section_parser('moleculetype', 'virtual_sitesn', context_type='block')
    def _vsn_interactions(self, line, lineno=0, context_type=''):

        context = self.current_block
        interaction_name = self.section[-1]
        delete = False

        tokens = collections.deque(_tokenize(line))

        n_atoms = self.interactions_natoms.get(interaction_name)

        _base_parser(
            tokens,
            context,
            context_type=context_type,
            section=interaction_name,
            natoms=n_atoms,
            delete=delete,
            current_meta=self.current_meta
            )
   
        # we need to reorder the atoms and interactions, because for virtual_sitesn
        # the function type is on section position followed by an undefined number
        # of atom indices from which the VS is built
 
        if '--' not in line:
           first_atom = context.interactions['virtual_sitesn'][-1].atoms[0]
           other_atoms = context.interactions['virtual_sitesn'][-1].atoms[2:]
           atoms = [first_atom] + other_atoms
   
           sec_atom = context.interactions['virtual_sitesn'][-1].atoms[1]
           params = [sec_atom]+context.interactions['virtual_sitesn'][-1].parameters
       
           context.interactions['virtual_sitesn'][-1].atoms[:]=atoms
           context.interactions['virtual_sitesn'][-1].parameters[:]=params


def _some_atoms_left(tokens, atoms, natoms):
    """
    Return True if the token list expected to contain atoms.

    If the number of atoms is known before hand, then the function compares the
    number of already found atoms to the expected number. If the '--' token if
    found, it is removed from the token list and there is no atom left.

    Parameters
    ----------
    tokens: collections.deque[str]
        Deque of token to inspect. The deque **can be modified** in place.
    atoms: list
        List of already found atoms.
    natoms: int or None
        The number of expected atoms if known, else None.

    Returns
    -------
    bool
    """
    if not tokens:
        return False
    if tokens and tokens[0] == '--':
        tokens.popleft()
        return False
    if natoms is not None and len(atoms) >= natoms:
        return False
    return True


def _parse_atom_attributes(token):
    """
    Parse bracketed tokens.

    Parameters
    ----------
    token: str
        Token in the form of a json dictionary.

    Returns
    -------
    dict
    """
    if not token.strip().startswith('{'):
        raise ValueError('The token should start with a curly bracket.')
    try:
        attributes = json.loads(token)
    except JSONDecodeError as error:
        raise ValueError('The following value is not a valid atom attribute token: "{}".'
                         .format(token)) from error
    modifications = {}
    for key, value in attributes.items():
        try:
            if '|' in value:
                modifications[key] = Choice(value.split('|'))
        except TypeError:
            pass
    attributes.update(modifications)
    return attributes


def _get_atoms(tokens, natoms):
    atoms = []
    while tokens and _some_atoms_left(tokens, atoms, natoms):
        token = tokens.popleft()
        if token.startswith('{'):
            msg = 'Found atom attributes without an atom reference.'
            raise IOError(msg)
        if tokens:
            next_token = tokens[0]
        else:
            next_token = ''
        if next_token.startswith('{'):
            atoms.append([token, _parse_atom_attributes(next_token)])
            tokens.popleft()
        else:
            atoms.append([token, {}])
    return atoms


def _treat_block_interaction_atoms(atoms, context, section):
    atom_names = list(context.nodes)
    all_references = []
    for atom in atoms:
        reference = atom[0]
        if reference.isdigit():
            # The indices in the file are 1-based
            reference = int(reference) - 1
            atom[0] = reference
        else:
            if reference not in context:
                msg = ('There is no atom "{}" defined in the block "{}". '
                       'Section "{}" cannot refer to it.')
                raise IOError(msg.format(reference, context.name, section))
            if reference[0] in '+-<>':
                msg = ('Atom names in blocks cannot be prefixed with + or -. '
                       'The name "{}", used in section "{}" of the block "{}" '
                       'is not valid in a block.')
                raise IOError(msg.format(reference, section, context.name))
        all_references.append(reference)
    return all_references


def _parse_interaction_parameters(tokens):
    parameters = []
    for token in tokens:
        if _is_param_effector(token):
            effector_name, effector_param_str = token.split('(', 1)
            effector_param_str = effector_param_str[:-1]  # Remove the closing parenthesis
            try:
                effector_class = PARAMETER_EFFECTORS[effector_name]
            except KeyError:
                raise IOError('{} is not a known parameter effector.'
                              .format(effector_name))
            if '|' in effector_param_str:
                effector_param_str, effector_format = effector_param_str.split('|')
            else:
                effector_format = None
            effector_param = [elem.strip() for elem in effector_param_str.split(',')]
            parameter = effector_class(effector_param, format_spec=effector_format)
        else:
            parameter = token
        parameters.append(parameter)
    return parameters


def _is_param_effector(token):
    return (
        '(' in token
        and not token.startswith('(')
        and token.endswith(')')
    )


def _base_parser(tokens, context, context_type, section, current_meta, natoms=None, delete=False):

    # Group the atoms and their attributes
    atoms = _get_atoms(tokens, natoms)

    if natoms is not None and len(atoms) != natoms:
        raise IOError('Found {} atoms while {} were expected.'
                      .format(len(atoms), natoms))

    # Normalize the atom references.
    # Blocks and links treat these references differently.
    # For blocks:
    # * references can be written as indices or atom names
    # * a reference cannot be prefixed by + or -
    # * an interaction cannot create a new atom
    # For links:
    # * references must be atom names, but they can be prefixed with one or
    #   more + or - to signify the order in the sequence
    # * interactions create nodes

    treated_atoms = _treat_block_interaction_atoms(atoms, context, section)

    # Everything that is not atoms are the interaction parameters    
    parameters = _parse_interaction_parameters(tokens)

    apply_to_all_interactions = context._apply_to_all_interactions[section]

    if current_meta:
       meta = {current_meta['condition']:current_meta['tag']}
    else:
       meta = {} #dict(collections.ChainMap(meta, apply_to_all_interactions))
 
    interaction = Interaction(
                  atoms=treated_atoms,
                  parameters=parameters,
                  meta=meta,
                  )

    interaction_list = context.interactions.get(section, [])
    interaction_list.append(interaction)
    context.interactions[section] = interaction_list


def _parse_block_atom(tokens, context):

    # deque does not support slicing
    first_six = (tokens.popleft() for _ in range(6))
    index, atype, resid, resname, name, charge_group = first_six
    
    if str(index) in context:
        msg = ('There is already an atom named "{}" in the block "{}". '
               'Atom names must be unique within a block.')
        raise IOError(msg.format(name, context.name))

    atom = {
        'atomname': name,
        'atype': atype,
        'resname': resname,
        'resid': int(resid),
        'charge_group': int(charge_group),
    }

    # charge and mass are optional, but charge has to be defined for mass to be
    if tokens:
        atom['charge'] = float(tokens.popleft())
    if tokens:
        atom['mass'] = float(tokens.popleft())

    attributes={}
    context.add_atom_from_index(dict(collections.ChainMap(attributes, atom)), index=index)

def read_itp(lines, force_field):
    director = ITPDirector(force_field)
    return list(director.parse(iter(lines)))
