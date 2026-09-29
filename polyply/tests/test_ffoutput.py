# Copyright 2024 Dr. Fabian Gruenewald
#
# Licensed under the PolyForm Noncommercial License 1.0.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://polyformproject.org/licenses/noncommercial/1.0.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from io import StringIO
from pathlib import Path
import pytest
import vermouth
from vermouth.ffinput import read_ff
from vermouth.molecule import Interaction
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
 #    'martini2',
     'parmbsc1',
     'martini3',
])
def test_ffoutput(tmp_path, libname):
    """
    Check if we can write and reread our own ff-libraries.
    """
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


def test_ffoutput_write_block_edges_false():
    """
    With write_block_edges=False, a block's [ edges ] directive must be
    omitted, but a link's own edges must still be written unconditionally.
    """
    force_field = vermouth.forcefield.ForceField("test")
    block = vermouth.molecule.Block(force_field=force_field)
    block.add_nodes_from([("BB", {"atype": "P1", "resid": 1, "resname": "A",
                                  "atomname": "BB", "charge_group": 1, "charge": 0.0}),
                          ("SC1", {"atype": "P2", "resid": 1, "resname": "A",
                                   "atomname": "SC1", "charge_group": 1, "charge": 0.0})])
    block.add_edge("BB", "SC1")
    block.nrexcl = 1
    force_field.blocks["A"] = block

    link = vermouth.molecule.Link()
    link.add_edge("BB", "+BB")
    force_field.links.append(link)

    stream = StringIO()
    ForceFieldDirectiveWriter(forcefield=force_field, stream=stream,
                              write_block_edges=False).write()
    text = stream.getvalue()

    moleculetype_section, link_section = text.split("[ link ]")
    assert "[ edges ]" not in moleculetype_section
    assert "[ edges ]" in link_section


def test_ffoutput_modifications():
    """
    A modification must be written such that it can be read back with
    all its PTM atoms, replace statements, edges, and interactions.
    """
    force_field = vermouth.forcefield.ForceField("test")
    modification = vermouth.molecule.Modification(name="SER-phos")
    modification.add_node("OG", **{"atomname": "OG", "element": "O",
                                   "resname": "SER", "PTM_atom": False,
                                   "replace": {"charge": -0.55}})
    modification.add_node("P", **{"atomname": "P", "element": "P",
                                  "atype": "opls_4", "charge": 1.2,
                                  "mass": 30.974, "PTM_atom": True})
    modification.add_edge("OG", "P")
    modification.interactions["bonds"].append(Interaction(atoms=["OG", "P"],
                                                          parameters=["1", "0.16", "900"],
                                                          meta={}))
    force_field.modifications["SER-phos"] = modification

    stream = StringIO()
    ForceFieldDirectiveWriter(forcefield=force_field, stream=stream).write()

    new_force_field = vermouth.forcefield.ForceField("test")
    read_ff(stream.getvalue().splitlines(keepends=True), new_force_field)

    assert list(new_force_field.modifications) == ["SER-phos"]
    new_modification = new_force_field.modifications["SER-phos"]
    # the parser sets the order attribute of every atom, which the writer
    # skips again, so it is not part of the comparison
    for name, attrs in modification.nodes(data=True):
        new_attrs = {attr: value for attr, value in new_modification.nodes[name].items()
                     if attr != "order"}
        assert new_attrs == attrs
    assert set(map(frozenset, new_modification.edges)) == set(map(frozenset, modification.edges))
    assert new_modification.interactions["bonds"] == modification.interactions["bonds"]
    # the resname tells which block the modification belongs to
    assert new_modification.nodes["OG"]["resname"] == "SER"
