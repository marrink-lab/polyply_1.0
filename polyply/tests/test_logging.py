import pytest
import logging
import numpy as np
import polyply
from vermouth.log_helpers import (StyleAdapter, BipolarFormatter,
                                  CountingHandler, TypeAdapter,)
from polyply.tests.test_build_file_parser import test_molecule, test_system

# Implement Logger
LOGGER = TypeAdapter(logging.getLogger('polyply'))
PRETTY_FORMATTER = logging.Formatter(fmt='{levelname:} - {type} - {message}',
                                     style='{')
DETAILED_FORMATTER = logging.Formatter(fmt='{levelname:} - {type} - {name} - {message}',
                                       style='{')
COUNTER = CountingHandler()

# Control above what level message we want to count
COUNTER.setLevel(logging.WARNING)

CONSOLE_HANDLER = logging.StreamHandler()
FORMATTER = BipolarFormatter(DETAILED_FORMATTER,
                             PRETTY_FORMATTER,
                             logging.DEBUG,
                             logger=LOGGER)

CONSOLE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(CONSOLE_HANDLER)
LOGGER.addHandler(COUNTER)

LOGGER = StyleAdapter(LOGGER)

loglevels = {0: logging.INFO, 1: logging.DEBUG, 2: 5}
LOGGER.setLevel(loglevels[1])


@pytest.mark.parametrize('_type, option, expected, warning, idxs', (
   # tag all nodes 1-4 that have name ALA, GLU, THR should not get tagged and raise warning
   ("cylinder",
    {"resname": "ALA", "start": 1, "stop": 9, "parameters":["in", np.array([5.0, 5.0, 5.0]), 5.0, 5.0]},
    [0, 1, 2, 3],
    "parsing build file: could not find resid {} with resname ALA in molecule.",
    range(5, 9),
    ),
   # raise warning that residue random cannot be found
   ("sphere",
    {"resname": "random", "start": 1, "stop": 3, "parameters":["in", np.array([10.0, 10.0, 10.0]), 5.0]},
    [],
    "parsing build file: could not find resid {} with resname random in molecule.",
    range(1, 4)),
   ))
def test_tag_nodes_logging(caplog, test_molecule, _type, option, expected, warning, idxs):
    polyply.src.build_file_parser.BuildDirector._tag_nodes(test_molecule, _type, option)
    for record, idx in zip(caplog.records, idxs):
        assert record.getMessage() == warning.format(idx)
    for node in test_molecule.nodes:
        if "restraints" in test_molecule.nodes[node]:
           assert node in expected

def test_mol_directive_logging(caplog, test_system):
    processor = polyply.src.build_file_parser.BuildDirector([], test_system)
    line = "AA 0 3"
    processor._molecule(line)
    assert caplog.records[0].getMessage() == "parsing build file: could not find molecule with name AA and index 2."
