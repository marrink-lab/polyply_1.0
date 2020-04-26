import os
import pathlib
import vermouth
from vermouth.ffinput import read_ff
from vermouth.gmx.rtp import read_rtp
from polyply import DATA_PATH
from .polyply_parser import read_polyply

FORCE_FIELD_PARSERS = {'rtp': read_rtp, 'ff': read_ff, 'itp': read_polyply}


def _resolve_lib_paths(lib_names, data_path):
    """
    select the appropiate files from data_path
    according to library names given.

    Parameters
    ----------
    lib_names: list[`os.path`]
        list of library names
    data_path:
        location of library files
    """
    files = []
    for name in lib_names:
        directory = os.path.join(data_path, name)
        for _file in os.listdir(directory):
            _file = os.path.join(directory, _file)
            files.append(_file)
    return files


def read_ff_from_files(paths, force_field):
    """
    read the input files for the defintion of blocks and links.

    Parameters
    ----------
    paths: list
           List of vaild file paths
    force_field: class:`vermouth.force_field.ForceField`

    Returns
    -------
    force_field: class:`vermouth.force_field.ForceField`
       updated forcefield

    """

    def wrapper(parser, path, force_field):
        with open(path, 'r') as file_:
            lines = file_.readlines()
            parser(lines, force_field=force_field)

    for path in paths:
        path = pathlib.Path(path)
        file_extension = path.suffix.casefold()[1:]

        try:
            parser = FORCE_FIELD_PARSERS[file_extension]
            wrapper(parser, path, force_field)
        except KeyError:
            raise IOError(
                "Cannot parse file with extension {}.".format(file_extension))

    return force_field


def load_library(name, lib_names, extra_files):
    """
    Load libraries and extra-files into vermouth
    force-field.

    Parameters
    ----------
    name: str
      Force-field name
    lib_names: list
      List of lirbary names
    extra_files: list
      List of extra files to include

    Returns
    -------
    :class:`vermouth.forcefield.Forcefield`
    """
    force_field = vermouth.forcefield.ForceField(name)
    if lib_names and extra_files:
        lib_files = _resolve_lib_paths(lib_names, DATA_PATH)
        all_files = lib_files + extra_files
    elif lib_names:
        all_files = _resolve_lib_paths(lib_names, DATA_PATH)
    elif extra_files:
        all_files = extra_files

    read_ff_from_files(all_files, force_field)
    return force_field
