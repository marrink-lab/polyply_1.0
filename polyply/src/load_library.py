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
from collections import ChainMap
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

def get_parser(file_path, file_parsers, is_lib_file):
    """
    check if file can be parsed and
    if possible return the respective parser

    Parameters
    ----------
    file_path: `pathlib.PosixPath`
        path to file
    file_parsers: dict
        dictionary of available file parsers
    is_lib_file: bool
        indicates whether the provided path is from a data library
    """
    file_extension = file_path.suffix[1:]
    if file_extension in file_parsers:
        return file_parsers[file_extension]
    elif not is_lib_file:
        msg = f"Cannot parse user provided file with extension {file_extension}."
        raise IOError(msg)
    elif file_extension not in ChainMap(FORCE_FIELD_PARSERS, BUILD_FILE_PARSERS):
        msg = f"File with unknown extension {file_extension} found in force field library."
        LOGGER.warning(msg)


def _resolve_lib_files(lib_names, data_path):
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
    paths: list[list[`pathlib.Path`], list[`pathlib.Path`]]
           The list contains exactly two sublist, respectively list containing
           the resolved library files and user provided files.
    storage_object: `polyply.src.topology.Topology` or `vermouth.forcefield.Forcefield`
    file_parsers: dict
        Dictionary of available file parsers

    Returns
    -------
    `polyply.src.topology.Topology` or `vermouth.forcefield.Forcefield`
    """

    def parse_file(parser, path, storage_object):
        with open(path, 'r') as file_:
            lines = file_.readlines()
            parser(lines, storage_object)

    lib_files, user_files = paths
    for path in user_files + lib_files or []:
        is_lib_file = path in lib_files
        parser = get_parser(path, file_parsers, is_lib_file)
        if parser:
            parse_file(parser, path, storage_object)
    return storage_object


def load_build_files(topology, lib_name, build_files):
    """
    Load build file options and molecule templates into topology.

    Parameters
    ----------
    topology: `polyply.src.topology`
    lib_name: `pathlib.Path`
        Library from where to load templates
    build_files: list[`pathlib.Path`]
        List of build files to parse

    Returns
    -------

    """
    lib_name = [lib_name] if lib_name else []
    all_files = [_resolve_lib_files(lib_name, DATA_PATH), build_files]
    read_options_from_files(all_files, topology, BUILD_FILE_PARSERS)


def load_ff_library(name, lib_names, extra_ff_files, force_field=None):
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
    `vermouth.forcefield.Forcefield`
    """
    if not force_field:
        force_field = vermouth.forcefield.ForceField(name)
    all_files = [_resolve_lib_files(lib_names, DATA_PATH), extra_ff_files]
    return read_options_from_files(all_files, force_field, FORCE_FIELD_PARSERS)
