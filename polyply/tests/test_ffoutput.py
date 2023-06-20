from pathlib import Path
import pytest
import vermouth
from vermouth.ffinput import read_ff
import polyply
from polyply.src.ffoutput import ForceFieldDirectiveWriter

def _read_force_field(fpath):
    """
    wrapper to read and return force-field
    """
    force_field = vermouth.forcefield.ForceField("test")
    with open(fpath, "r") as _file:
        lines = _file.readlines()
    read_ff(lines, force_field)
    return force_field

def equal_blocks(block1, block2):
    """
    Need to overwrite since obviously
    the force-fields cannot be the same.
    """
    return (block1.nrexcl == block2.nrexcl and
            block1.same_nodes(block2) and
            block1.same_edges(block2) and
            block1.same_interactions(block2) and
            block1.name == block2.name )

def compare_patterns(patterns1, patterns2):
    """
    Patterns are evil so we also need a
    special compare function.
    """
    assert len(patterns1) == len(patterns2)
    for pattern1, pattern2 in zip(patterns1, patterns2):
        for entry1, entry2 in zip(pattern1, pattern2):
            assert entry1[0] == entry2[0]
            assert not vermouth.utils.are_different(entry1[1],
                                                    entry2[1])
    return True

def equal_links(link1, link2):
    """
    Needs to overwrite for the same reason
    as for blocks.
    """
    return (equal_blocks(link1, link2)
           and link1.same_non_edges(link2)
           and link1.removed_interactions == link2.removed_interactions
           and link1.molecule_meta == link2.molecule_meta
           and compare_patterns(link1.patterns, link2.patterns)
           and set(link1.features) == set(link2.features)
           )

def equal_ffs(ff1, ff2):
    """
    Compare two forcefields.
    """
    assert len(ff1.blocks) == len(ff2.blocks)
    # compare blocks
    for name, block in ff1.blocks.items():
        assert equal_blocks(block, ff2.blocks[name])

    for link1, link2 in zip(ff1.links, ff2.links):
        assert equal_links(link1, link2)
    return True

@pytest.mark.parametrize("libname", [
     '2016H66',
     'gromos53A6',
     'oplsaaLigParGen',
     'martini2',
     'parmbsc1',
])
def test_ffoutput(tmp_path, libname):
    """
    Check if we can write and reread our own ff-libraries.
    """
    tmp_path = "/coarse/fabian/current-projects/polymer_itp_builder/polyply_2.0/polyply/tests/test_data/tmp"
    lib_path = Path(polyply.DATA_PATH) / libname
    for idx, _file in enumerate(lib_path.iterdir()):
        if _file.suffix == ".ff":
            # read the forcefield
            force_field = _read_force_field(_file)
            # write the forcefield
            tmp_file = Path(tmp_path) / (str(idx) + f"{libname}_new.ff")
            with open(tmp_file, "w") as filehandle:
                ForceFieldDirectiveWriter(forcefield=force_field, stream=filehandle).write()
            # read the smae forcefield file
            force_field_target = _read_force_field(tmp_file)
            assert equal_ffs(force_field, force_field_target)
