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
import os
import copy
from itertools import zip_longest
from vermouth.parser_utils import SectionLineParser
from vermouth.molecule import Interaction
from vermouth.gmx.itp_read import read_itp
from .meta_molecule import MetaMolecule, _make_edges
from tqdm import tqdm

class TOPDirector(SectionLineParser):

    COMMENT_CHAR = ';'

    atom_idxs = {'bonds': [0, 1],
                 'bondtypes':[0, 1],
                 'position_restraints': [0],
                 'angles': [0, 1, 2],
                 'angletypes':[0, 1, 2],
                 'constraints': [0, 1],
                 'constrainttypes': [0, 1],
                 'dihedrals': [0, 1, 2, 3],
                 'dihedraltypes': [0, 1, 2, 3],
                 'pairs': [0, 1],
                 'pairtypes': [0, 1],
                 'exclusions': [slice(None, None)],
                 'virtual_sitesn': [0, slice(2, None)],
                 'virtual_sites1': [0],
                 'virtual_sites2': [0, 1, 2, 3],
                 'virtual_sites3': [0, 1, 2, 3],
                 'virtual_sites4': [slice(0, 5)],
                 'pairs_nb': [0, 1],
                 'settles': [0],
                 'distance_restraints':  [0, 1],
                 'dihedral_restraints':  [slice(0, 4)],
                 'orientation_restraints': [0, 1],
                 'angle_restraints': [slice(0, 4)],
                 'angle_restraints_z': [0, 1]}

    def __init__(self, topology, cwdir=None):
        super().__init__()
        self.force_field = topology.force_field
        self.topology = topology
        self.current_meta = None
        self.current_itp = None
        self.itp_lines = []
        self.molecules = []
        self.cwdir = cwdir
        self.header_actions = {
            ('moleculetype',): self._new_itp
        }
        self.pragma_actions = {
            '#define': self.parse_define,
            '#include': self.parse_include
        }

    def dispatch(self, line):
        """
        Looks at `line` to see what kind of line it is, and returns either
        :meth:`parse_header` if `line` is a section header or
        :meth:`vermouth.parser_utils.SectionLineParser.parse_section` otherwise.
        Calls :meth:`vermouth.parser_utils.SectionLineParser.is_section_header` to see
        whether `line` is a section header or not.

        Parameters
        ----------
        line: str

        Returns
        -------
        collections.abc.Callable
            The method that should be used to parse `line`.
        """
        if self.is_pragma(line):
            return self.parse_top_pragma
        elif self.is_star_comment(line):
            return self._skip
        else:
            return super().dispatch(line)

    @staticmethod
    def is_pragma(line):
        """
        Parameters
        ----------
        line: str
            A line of text.

        Returns
        -------
        bool
            ``True`` if `line` is a def statement.
        """
        return line.startswith('#')

    @staticmethod
    def is_star_comment(line):
        """
        Star comments are special comments
        usually found at the beginning of GMX
        library topology files. They are different
        from the regular comment as they are specific
        to top library files.

        Parameters
        ----------
        line: str
            A line of text.

        Returns
        -------
        bool
            ``True`` if `line` is a star comment.
        """
        return line.startswith('*')

    def parse_top_pragma(self, line, lineno=0):
        """
        Parses the beginning and end of define sections
        with line number `lineno`. Sets attr current_meta
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
        if line == '#endif':
            if self.current_itp:
                self.current_itp.append(line)
            elif self.current_meta is None:
                raise IOError("Your #ifdef section is orderd incorrectly."
                              "At line {} I read {} but I haven not read"
                              "an #ifdef before.".format(lineno, line))
            else:
               self.current_meta = None

        elif line.startswith("#else"):
            if self.current_itp:
                self.current_itp.append(line)
            elif self.current_meta is None:
               raise IOError("Your #ifdef section is orderd incorrectly."
                             "At line {} I read {} but I haven not read"
                             "a ifdef before.".format(lineno, line))
            else:
               inverse = {"ifdef": "ifndef", "ifndef": "ifdef"}
               tag = self.current_meta["tag"]
               condition = inverse[self.current_meta["condition"]]
               self.current_meta = {'tag': tag, 'condition': condition}

        elif line.startswith("#ifdef") or line.startswith("#ifndef"):
            if self.current_itp:
                self.current_itp.append(line)
            elif self.current_meta is None:
                condition, tag = line.split()
                self.current_meta = {'tag': tag,
                                     'condition': condition.replace("#", "")}
            elif self.current_meta is not None:
                raise IOError("Your {} section is orderd incorrectly."
                              "At line {} I read {} but there is still"
                              "an open #ifdef/#ifndef section from"
                              "before.".format(self.current_meta['tag'], lineno, line.split()[0]))

        elif line.split()[0] in self.pragma_actions:
            action = self.pragma_actions[line.split()[0]]
            action(line)
        else:
            raise IOError("Don't know how to parse pargma {} at"
                          "line {}.".format(line, lineno))

    def parse_header(self, line, lineno=0):
        """
        Parses a section header with line number `lineno`. Sets
        :attr:`vermouth.parser_utils.SectionLineParser.section`
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
        result = super().parse_header(line, lineno)
        action = self.header_actions.get(tuple(self.section))
        if action:
            action()

        if self.current_itp is not None:
            self.current_itp.append(line)
        return result

    def finalize(self, lineno=0):
        """
        Called at the end of the file and checks that all pragmas are closed
        before calling the parent method.
        """
        if self.current_itp:
            self.itp_lines.append(self.current_itp)

        if self.current_meta is not None:
            raise IOError("Your {} section is orderd incorrectly."
                          "There is no #endif for this pragma.".format(self.current_meta))

        for lines in self.itp_lines:
            read_itp(lines, self.force_field)

        total_count = 0
        _make_edges(self.force_field)

        for mol_name, n_mol in self.molecules:
            block = self.force_field.blocks[mol_name]
            graph = MetaMolecule._block_graph_to_res_graph(block)
            for idx in range(0, int(n_mol)):
                graph_copy = graph.copy(as_view=False)
                new_mol = MetaMolecule(graph_copy,
                                       force_field=self.force_field,
                                       mol_name=mol_name)
                new_mol.molecule = self.force_field.blocks[mol_name].to_molecule()
                new_mol.mol_name = mol_name

                self.topology.add_molecule(new_mol)
                self.topology.mol_idx_by_name[mol_name].append(total_count)
                total_count += 1
        super().finalize()

    def _new_itp(self):
        if self.current_itp:
           self.itp_lines.append(self.current_itp)
        self.current_itp = []

    @SectionLineParser.section_parser('system')
    def _system(self, line, lineno=0):
        """
        Parses the lines in the '[system]'
        directive and stores it.
        """
        system_lines = self.topology.description
        system_lines.append(line)
        self.description = system_lines

    @SectionLineParser.section_parser('molecules')
    def _molecules(self, line, lineno=0):
        """
        Parses the lines in the '[molecules]'
        directive and stores it.
        """
        # we need to keep the order here so cannot make it a dict
        # also mol names do not need to be unique
        name, n_mol = line.split()
        self.molecules.append((name, n_mol))

    @SectionLineParser.section_parser('defaults')
    def _defaults(self, line, lineno=0):
        """
        Parse and store the defaults section.
        """
        defaults = ["nbfunc", "comb-rule", "gen-pairs", "fudgeLJ", "fudgeQQ"]
        numbered_terms = ["nbfunc", "comb-rule", "fudgeLJ", "fudgeQQ"]
        tokens = line.split()

        # Parse all defaults to a dict up to the last default metioned
        # Note that gen_pairs, fudgeLJ and fudgeQQ not need to be set
        self.topology.defaults = dict(zip(defaults[0:len(tokens)], tokens))

        # we cannot interpret Buckingham Potentials at the moment so we crash
        if self.topology.defaults["nbfunc"] == "2":
           raise IOError("Buckingham Potential requested but this potential form"
                         "currently is not implemented.")

        # converts the defaults that are numbers to numbers
        # we need to parse them first because they are not guaranteed to be provided
        for token_name in numbered_terms:
            if token_name in self.topology.defaults:
                 self.topology.defaults[token_name] = float(self.topology.defaults[token_name])

        # sets gen-pairs to no when it is not provied
        if "gen-pairs" not in self.topology.defaults:
            self.topology.defaults["gen-pairs"] = "no"

    @SectionLineParser.section_parser('atomtypes')
    def _atomtypes(self, line, lineno=0):
        """
        Parse and store atomtypes section
        """
        tokens = line.split()
        atom_name = tokens.pop(0)
        tokens.reverse()
        atom_type_line = dict(zip_longest(["nb2", "nb1", "ptype",
                                           "charge", "mass",
                                           "atom_num", "bond_type"], tokens, fillvalue=None))
        floats = ["nb1", "nb2", "charge", "mass", "atom_num"]
        for term, value in atom_type_line.items():
             if term in floats and value:
                 atom_type_line[term] = float(value)

        if None in atom_type_line:
            msg = ("Can't parse line {}. Found more parameters than expected.")
            raise OSError(msg.format(line))

        self.topology.atom_types[atom_name] = atom_type_line

    @SectionLineParser.section_parser('nonbond_params')
    def _nonbond_params(self, line, lineno=0):
        """
        Parse and store nonbond params
        """
        atom_1, atom_2, func, nb1, nb2 = line.split()
        self.topology.nonbond_params[frozenset([atom_1, atom_2])] = {"f": int(func),
                                                                     "nb1": float(nb1),
                                                                     "nb2": float(nb2)}
    @SectionLineParser.section_parser('pairtypes')
    @SectionLineParser.section_parser('angletypes')
    @SectionLineParser.section_parser('dihedraltypes')
    @SectionLineParser.section_parser('bondtypes')
    @SectionLineParser.section_parser('constrainttypes')
    def _type_params(self, line, lineno=0):
        """
        Parse and store bonded types
        """
        section_name = self.section[-1]
        inter_type = section_name[:-5] + "s"
        atoms, params = self._split_atoms_and_parameters(line.split(),
                                                         self.atom_idxs[section_name])

        self.topology.types[inter_type][tuple(atoms)].append((params, self.current_meta))


    @SectionLineParser.section_parser('implicit_genborn_params')
    @SectionLineParser.section_parser('cmaptypes')
    def _skip(self, line, lineno=0):
        pass

    @SectionLineParser.section_parser('moleculetype')
    @SectionLineParser.section_parser('moleculetype', 'atoms')
    @SectionLineParser.section_parser('moleculetype', 'bonds')
    @SectionLineParser.section_parser('moleculetype', 'angles')
    @SectionLineParser.section_parser('moleculetype', 'dihedrals')
    @SectionLineParser.section_parser('moleculetype', 'impropers')
    @SectionLineParser.section_parser('moleculetype', 'constraints')
    @SectionLineParser.section_parser('moleculetype', 'pairs')
    @SectionLineParser.section_parser('moleculetype', 'exclusions')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites1')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites2')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites3')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites4')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sitesn')
    @SectionLineParser.section_parser('moleculetype', 'position_restraints')
    @SectionLineParser.section_parser('moleculetype', 'pairs_nb')
    @SectionLineParser.section_parser('moleculetype', 'settles')
    @SectionLineParser.section_parser('moleculetype', 'distance_restraints')
    @SectionLineParser.section_parser('moleculetype', 'orientation_restraints')
    @SectionLineParser.section_parser('moleculetype', 'dihedral_restraints')
    @SectionLineParser.section_parser('moleculetype', 'angle_restraints')
    @SectionLineParser.section_parser('moleculetype', 'angle_restraints_z')
    def _molecule(self, line, lineno=0):
        """
        Parses the lines of the [atoms] directive.
        """
        self.current_itp.append(line)

    def parse_define(self, line):
        """
        Parse define statemetns
        """
        tokens = line.split()

        if len(tokens) > 2:
            tag = line.split()[1]
            parameters = line.split()[2:]
        else:
            _, tag = line.split()
            parameters = True

        definition = {tag: parameters}
        self.topology.defines.update(definition)

    def parse_include(self, line):
        """
        parse include statemnts
        """
        path = line.split()[1].strip('\"')
        if self.current_meta:
           # the current file is between ifdef
           # however tag is not in defines we have
           # read so the file is not read
           if self.current_meta["condition"] == "ifdef"\
              and self.current_meta["tag"] not in self.topology.defines:
                 return
           # the current file is between ifndef
           # so if tag is defined we ignore this file
           elif self.current_meta["condition"] == "ifndef"\
              and self.current_meta["tag"] in self.topology.defines:
                 return

        if self.cwdir:
           filename = os.path.join(self.cwdir, path)
           cwdir = os.path.dirname(filename)
        else:
           cwdir = os.path.dirname(path)
           filename = path

        if not os.path.exists(filename):
            msg = ("Cannot find file {}. This can happen when you "
                  "1) typed the path to the file wrongly or 2) when you "
                  "try to include force-field files from the GMX "
                  "library (e.g. #include \"gromos\"). Instead provide "
                  "the full path. Another source for this error can be "
                  "that you have a #ifdef section with an #include but "
                  "your include file does not exist. In that case if you "
                  "don't have the file remove the #include statement.")
            raise IOError(msg.format(filename))

        with open(filename, 'r') as _file:
            lines = _file.readlines()

        read_topology(lines, topology=self.topology, cwdir=cwdir)

    def _split_atoms_and_parameters(self, tokens, atom_idxs):
        """
        Returns atoms from line based on the indices defined in `atom_idxs`.
        It also interprets slices etc. stored as strings.

        Parameters:
        ------------
        tokens: collections.deque[str]
            Deque of token to inspect. The deque **can be modified** in place.
        atom_idxs: list of ints or strings that are valid python slices

        Returns:
        -----------
        list
        """

        atoms = []
        remove = []
        # first we extract the atoms from the indices given using
        # ints or slices
        for idx in atom_idxs:
            if isinstance(idx, int):
                atoms.append(tokens[idx])
                remove.append(idx)
            elif isinstance(idx, slice):
                atoms.extend(tokens[idx])
                idx_range = range(0, len(tokens))
                remove += idx_range[idx]
            else:
                raise IOError

        # everything that is left are parameters, which we
        # get by simply deleting the atoms from tokens

        for index in sorted(remove, reverse=True):
            del tokens[index]

        return atoms, tokens


def read_topology(lines, topology, cwdir=None):
    """
    Parses `lines` of itp format and adds the
    molecule as a block to `force_field`.

    Parameters
    ----------
    lines: list
        list of lines of an itp file
    force_field: :class:`vermouth.forcefield.ForceField`
    """
    director = TOPDirector(topology, cwdir)
    return list(director.parse(iter(lines)))
