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
from pathlib import Path
import vermouth
from vermouth.gmx.rtp import read_rtp
from vermouth.citation_parser import read_bib
from vermouth.log_helpers import StyleAdapter, get_logger
from polyply import DATA_PATH
from .ff_parser_sub import read_ff
from .build_file_parser import read_build_file
from .polyply_parser import read_polyply

LOGGER = StyleAdapter(get_logger(__name__))
BUILD_FILE_PARSERS = {'bld': read_build_file}
FORCE_FIELD_PARSERS = {'rtp': read_rtp, 'ff': read_ff, 'itp': read_polyply, 'bib': read_bib}

def determine_parser(file_, file_parsers):
    """
    check if file can be parsed and
    if possible return the respective parser

    Parameters
    ----------
    file_: class:`pathlib.PosixPath`
        path to file
    file_parsers: dict
        dictionary of available file parsers
    """
    file_extension = file_.suffix[1:]
    if file_extension not in file_parsers:
        msg = "Cannot parse file file with extension {}".format(file_extension)
        raise IOError(msg)
    else:
        return file_parsers[file_extension]


def _resolve_lib_paths(lib_names, data_path):
    """
    select the appropiate files from a file path
    according to library names given.

    Parameters
    ----------
    lib_names: list[`pathlib.Path`]
        list of library names
    data_path:
        location of library files
    """
    files = []
    data_path = Path(data_path)
    for name in lib_names or []:
        directory = data_path.joinpath(name)
        for file_ in os.listdir(directory):
            file_path = directory.joinpath(file_)
            files.append(file_path)
    return files


def read_options_from_files(paths, storage_object, file_parsers):
    """
    read the input files for the definition of blocks and links.

    Parameters
    ----------
    paths: list[`pathlib.Path`]
           List of provided file paths
    storage_object: topology or forcefield
    file_parsers: dict
        dictionary of available file parsers

    """

    def parse_file(parser, path, storage_object):
        with open(path, 'r') as file_:
            lines = file_.readlines()
            parser(lines, storage_object)

    for path in paths or []:
        parser = determine_parser(path, file_parsers)

        parse_file(parser, path, storage_object)


def load_build_files(topology, lib_names, build_file):
    """
    Load build file options and molecule templates into topology.

    Parameters
    ----------
    topology: :class:`polyply.src.topology`
    build_file: str
        List of build files to parse
    lib_names: list[str]
        List of library names for which to load templates

    Returns
    -------

    """
    all_files = _resolve_lib_paths(lib_names, DATA_PATH)
    all_files.extend(build_file)
    read_options_from_files(all_files, topology, BUILD_FILE_PARSERS)


def load_ff_library(name, lib_names, extra_ff_file):
    """
    Load libraries and extra-files into vermouth
    force-field.

    Parameters
    ----------
    name: str
      Force-field name
    lib_names: list[`pathlib.Path`]
      List of library names
    extra_files: list[`pathlib.Path`]
      List of extra files to include

    Returns
    -------
    :class:`vermouth.forcefield.Forcefield`
    """
    force_field = vermouth.forcefield.ForceField(name)
    all_files = _resolve_lib_paths(lib_names, DATA_PATH)
    all_files.extend(extra_ff_file)
    read_options_from_files(all_files, force_field, FORCE_FIELD_PARSERS)
    return force_field
