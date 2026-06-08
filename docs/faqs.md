# FAQs

Here is a list of frequently asked questions. If you cannot find your question here, simply
[raise an issue](https://github.com/marrink-lab/polyply_1.0/issues) for further help. If
you can find your issue here but cannot resolve it using the answers, raise an issue with
the tag `FAQ`.

### Why is my structure generation slow?

It could be that the target density is too high for efficient packing or the random-walk
gets trapped. Try decreasing the density or increasing the maximum force (`-mf`).

### Why does my structure generation crash with a KeyError?

This happens most likely in one of two scenarios:

1. You have specified an atomtype in your itp which is not defined in the topology itp
   file; the program crashes at the template generation stage.
2. Your residue names are not unique; in this case the program crashes during the
   backmapping steps. All residue names have to be unique.

### Why does `polyply gen_params` fail with "Your molecule consist of disconnected parts"?

This means that you have parts in your molecule that are not connected by bonds, angles,
dihedrals, or improper dihedrals. Typically this means that your links do not apply (see
below). But it can also happen when, for example, you have a virtual site which is not
connected to other atoms by bonds, angles, and so on. In this case, to avoid the warning,
you need to add a new directive called `[edges]`, where you specify pairs of atom names of
the virtual-site and its constructing atoms. For example, for atoms A, B, C and
virtual-site D:

```
[ virtual_sitesn ]
D A B C -- 1
[ edges ]
A D
B D
C D
```

### Why is the function type of my virtual-site not preserved?

In the ff-format the virtual-sites follow a special syntax that is different from the
itp-file syntax. In the ff-format the first atom is the virtual-site, followed by the
constructing atoms, followed by a double dash (`--`) and the function type. See the
previous question.

### Why do I get infinities in the energy minimization?

This means that after backmapping to target resolution two beads/atoms are overlapping. To
resolve this problem, try to create a system at lower target density, reduce the maximum
force allowed in the random walk (i.e. set a lower value using `-mf`), or try to minimize
using thermodynamic integration.

### Why does my link not apply?

There are many reasons, but typically it is because the algorithm cannot find matching
atoms, or the link at residue level does not match any fragment in the target graph. To
resolve this problem, run `gen_itp` with the `-vv` flag. This will print why links don't
apply. Another typically encountered problem is that for exclusions, pairs, virtual-sites,
and improper dihedral angles you will have to provide an `[edges]` directive describing
what the fragment looks like that these interactions correspond to. In this case consult
the [tutorial on writing ff-files](reference/writing-ff-input-files.md) for more
information.

### I cannot run the package because of a 'wrapper' error?

This typically occurs because the numba version installed on the machine is too old. Try
reinstalling numba to the most recent version.

### How do I include a starting structure?

Starting structures can be provided with the `-c` option. Note that the starting structure
coordinates are associated to the first N atoms in your topology file. If you want to extend
some existing molecules by polymers, use the `-res` flag to tell polyply that all residues
with the name mentioned there have to be built. In that case all other coordinates are
considered provided with the `-c` option. Again, the order in the coordinate file needs to
match the order of all other coordinates in the topology file.

### How do I install polyply?

We recommend installing polyply in a virtual environment. In that case simply create a new
environment and use `pip install polyply`. If you are working with conda, check out the
conda install instructions. Note that installing numba with polyply gives great speed-ups.
See the [Installation](installation.md) guide for details.

### How do I install the development version?

To install the development version use:

```bash
pip install git+https://github.com/marrink-lab/polyply_1.0.git#polyply_1.0
```

### How does polyply deal with chirality in structure generation?

Polyply does not specifically take care to create residues with the appropriate chirality.
Instead the chirality should be enforced by including an appropriate improper dihedral. That
dihedral angle only has to be present for the energy minimization step to push the
appropriate conformation, unless the force-field defines such a dihedral angle by default.
For example, in GROMOS PMA the chiral hydrogen is missing, so within GROMOS an improper
dihedral angle is utilized by default to set the chirality. In contrast, for PMMA that is
not the case, because the methyl group is represented explicitly. In this case we have
implemented such a dihedral angle in the library, and it is switched on by setting the
`eq_polyply` define statement in the mdp file. Note that the user must use this setting at
least in the energy minimization to ensure the expected chirality.

### Why does `polyply gen_coords` error when I include a GROMACS default force-field?

Polyply is not able to find the directory where the GROMACS default force-fields are
located. If you provide the full path to the directory it should work.

### Why does `polyply gen_coords` crash with multiple Martini ions?

This is a problem of the martini2/3 ion definitions and is easily fixed. Simply change the
default residue name `ION` to something different for all ion species in the system. All
residues need to have a unique residue name.
