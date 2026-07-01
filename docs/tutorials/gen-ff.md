# Tutorial: Generating an arbitrary force field for a polymer

## Introduction

This guide explains how to generate an OPLS-AA atomistic force field for a polymer,
using [LigParGen](https://zarbi.chem.yale.edu/ligpargen/) and
[`polyply gen_ff`](https://github.com/marrink-lab/polyply_1.0).

!!! warning "`gen_ff` is not in a stable release yet"

    This workflow relies on `polyply gen_ff`, which currently lives only on the
    experimental **`gen_ff_clean`** development branch (see the installation step
    below). Expect experimental features and potential breaking changes, and be
    aware that commands and flags shown here may change before `gen_ff` is
    included in an official polyply release.

**Why OPLS-AA?** While it is a relatively widely used force field for non-biological
organic molecules, the choice of OPLS-AA here is completely arbitrary and the workflow is force-field agnostic.
It could have been GAFF2, OpenFF, etc.

**What is SMILES?** SMILES is a text notation for chemical structures.
You can find a [notation explainer here](https://matlantis.com/en/resources/blog/how-to-write-smiles/)
and visualize structures [here](https://pubchem.ncbi.nlm.nih.gov/edit3/index.html).

---

## Prerequisites

- A SMILES string or PDB file for your pentamer
- [UV](https://docs.astral.sh/uv/#highlights) or Conda installed
- Python 3.x

---

## 1. LigParGen

We use the [LigParGen server](https://zarbi.chem.yale.edu/ligpargen/) to generate
an initial force field in OPLS-AA format from a small-molecule fragment of the polymer.

### The 200-atom limit and the trimmed-fragment strategy

LigParGen only supports molecules of up to 200 atoms. For polymers, this means
we cannot upload the full chain. However, we need at least 3–4 repeating units to
correctly capture the dihedral environment of the central repeat unit.

The solution is to upload a **trimmed fragment**: one complete central repeating unit
flanked by 2 units on each side. This captures the correct dihedral environment
without exceeding the atom limit. For polymers with short repeat units (e.g. PS,
see Example 1), a full 5-mer fits comfortably within the 200-atom limit.
See special cases below if a 5-mer exceeds the 200-atom limit.

For example, the SMILES string below represents a 5-mer PS chain: 
```
CCC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C
```
Note that the PS monomer is `CC(c1ccccc1)` and the string is terminated on both sides
by two methyl gropus (`C`).


### Generating the force field

1. Paste your SMILES string into the LigParGen input field.
2. Under **charge model**, select **(neutral or charged molecules)**.
3. Click **Submit Molecule**.
4. Download:
   - **TOP for Gromacs** (`.itp`) — force field topology
   - **PDB** — 3D structure

### Structure of the ITP file

The ITP file contains several sections. The most important for our purposes is
**`[ atomtypes ]`**, which defines atom type names and their associated
Lennard-Jones parameters (σ and ε).

---

## 2. Installation

We need two environments: one for `gen_ff` (a development branch of polyply used
to generate the `.ff` file) and one for the stable polyply release (used for all
subsequent polyply steps such as `gen_params` and `gen_coords`).

**With UV** (recommended):
```bash
uv venv gen_ff --python 3.11
source gen_ff/bin/activate 
uv pip install git+https://github.com/gruenewald-lab/CGsmiles.git
uv pip install scipy matplotlib shapely pytest
git clone git@github.com:marrink-lab/polyply_1.0.git
cd polyply_1.0
git checkout gen_ff_clean            # development branch
uv pip install -e .
```

**With Conda**:
```bash
conda create -n gen_ff python=3.11
conda activate gen_ff
pip install git+https://github.com/gruenewald-lab/CGsmiles.git
pip install scipy matplotlib shapely pytest
git clone git@github.com:marrink-lab/polyply_1.0.git
cd polyply_1.0
git checkout gen_ff_clean            # development branch
pip install -e .
```

!!! note

    `gen_ff_clean` is a development branch — expect experimental features and
    potential breaking changes (see the warning at the top of this page).

For the stable polyply environment (needed for validation and all downstream steps):
```bash
# UV
uv venv polyply-stable --python 3.11
source polyply-stable/bin/activate
uv pip install polyply

# Conda
conda create -n polyply-stable python=3.11
conda activate polyply-stable
pip install polyply
```

---

## 3. Atom type relabeling

LigParGen assigns its own internal atom type names rather than the standard OPLS-AA
labels that GROMACS expects. The script `atomtype_reduction_oplsLigParGen.py` fixes
this: it matches each atom type's Lennard-Jones parameters against a dictionary of
known OPLS-AA types and rewrites the topology with the correct labels,
outputting `system_OK.top`.

Both scripts ship with this documentation, under `docs/tutorials/ff_example/` in the
polyply repository you already cloned above — `atomtype_reduction_oplsLigParGen.py`
(available [here](./ff_example/atomtype_reduction_oplsLigParGen.py)) together with its
helper `itp_handling.py`, which must sit in a `src/` subfolder relative to it (as it does
in the repository).

```bash
./atomtype_reduction_oplsLigParGen.py -f <your_file>_LigParGen.itp
mv system_OK.top <your_file>_LigParGen_OK.itp
```

Then remove or comment out the entire `[ atomtypes ]` section from the `_OK.itp`
and save as `_clean.itp`. This section is stripped because GROMACS expects atomtypes
to be defined once at the system level (in the `.top` file), not repeated in each `.itp`.

---

## 4. CGsmiles string

CGsmiles describes polymer topology at two levels. For example, for PS:

```
{[#CH3][#PS]|5[#CH3]}.{#CH3=C[>][<],#PS=[>]CC(c1ccccc1)[<]}
```

- The first `{...}` block is the **polymer-level** graph: repeat units and their
  connectivity (`|5` means 5 repetitions of `#PS`)
- The second `{...}` block is the **monomer-level** graph: each unit defined by its
  SMILES string, with `[>]` and `[<]` denoting inter-unit bond attachment points

See the [CGsmiles documentation](https://cgsmiles.readthedocs.io/en/latest/gettingstarted/syntax_examples.html)
and [publication](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00064) for full syntax details.

---

## 5. Running `polyply gen_ff`

```bash
source gen_ff/bin/activate   # or: conda activate gen_ff

polyply gen_ff -i <your_file>_LigParGen_clean.itp \
               -s "<CGsmiles string>" \
               -c <monomer1>:0 <monomer2>:0 ... \
               -o <polymer>.ff
```

Key flags:
- `-i` — input `.itp` file (**use the `_clean.itp` version**)
- `-s` — CGsmiles string
- `-c` — charge per monomer residue (e.g. `PS:0` for neutral)
- `-o` — output `.ff` filename

---

## Examples

### Example 1: Polystyrene (PS)

PS is a simple vinyl polymer with a styrene repeat unit. Its repeat unit is small
enough that a full 5-mer fits within LigParGen's 200-atom limit.
This makes it a clean, minimal case to get familiar with the workflow.

**SMILES for LigParGen** (5-mer with methyl terminals):
```
CCC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C
```

Download and relabel atomtypes as described in Sections 1 and 3, saving the
final cleaned file as `PS_n5_LigParGen_clean.itp`.

**CGsmiles string:**
```
{[#CH3][#PS]|5[#CH3]}.{#CH3=C[>][<],#PS=[>]CC(c1ccccc1)[<]}
```

**Running `gen_ff`:**
```bash
source gen_ff/bin/activate

polyply gen_ff -i PS_n5_LigParGen_clean.itp \
               -s "{[#CH3][#PS]|5[#CH3]}.{#CH3=C[>][<],#PS=[>]CC(c1ccccc1)[<]}" \
               -c CH3:0 PS:0 -o PS.ff
```

**Validation:**

To generate coordinates and run a quick energy minimization you need two more input
files, both provided alongside this tutorial in `docs/tutorials/ff_example/`:

- [`system_test.top`](./ff_example/system_test.top) — the system topology. It includes
  the OPLS-AA force field and the itp generated by `gen_params` below (`PS_n5_test.itp`),
  and lists a single `PS` molecule.
- [`em.mdp`](./ff_example/em.mdp) — a standard steepest-descent energy-minimization
  parameter file.

!!! warning "Set the OPLS-AA force-field path in `system_test.top`"

    `system_test.top` includes the OPLS-AA force field with a placeholder path:

    ```
    #include "/absolute-path-to/share/gromacs/top/oplsaa.ff/forcefield.itp"
    ```

    `gen_coords` resolves this include using the **absolute path**, so it cannot be
    known in advance — it depends on your GROMACS installation. Before running the
    commands below, replace `/absolute-path-to/` with the actual location of your
    GROMACS `top` directory (for example `/usr/local/gromacs/share/gromacs/top/...`;
    if `gmx` is on your `PATH`, the command `$(dirname $(which gmx))/../share/gromacs/top`
    resolves to it). A relative `#include "oplsaa.ff/forcefield.itp"` is **not**
    sufficient here.

```bash
source polyply-stable/bin/activate   # or: conda activate polyply-stable

polyply gen_params -f PS.ff -seq CH3:1 PS:5 CH3:1 -o PS_n5_test.itp -name PS
polyply gen_coords -p system_test.top -o system_test.gro -name PS -box 5 5 5
gmx grompp -p system_test.top -c system_test.gro -f em.mdp -o 1-em
gmx mdrun -v -deffnm 1-em
```

