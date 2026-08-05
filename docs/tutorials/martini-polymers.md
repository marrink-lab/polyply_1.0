# Tutorial: Martini Polymers

## 1 - Introduction

Polyply facilitates the easy generation of polymer melts / amorphous blends, polymer
solutions, and even liquid-liquid phase separated systems. In this tutorial, we are going
to use polyply to generate itp files and starting configurations for these three cases. In
particular, you will generate an amorphous polymer simulation using pure PEO and a blend of
two other polymers. Subsequently, we turn to polymer–solvent systems and last but not least
liquid-liquid phase separated systems of Dextran and PEO. Note that all basic steps apply
in the same manner to all-atom models, but make sure to also check out the
[GROMOS tutorial](gromos-polymer-melts.md) for all-atom simulations.

## 2 - Generating an itp file

First, we need to generate an itp file for the polymers that we want to study. Polyply is
shipped with a range of polymer input files collected in libraries, which can be used to
generate itp files for polymers. Note that you can also implement your own polymer
definition — to do so, check out the tutorial on
[writing .ff input files](../reference/writing-ff-input-files.md). If you have any trouble,
always feel free to raise an issue. To inspect which libraries are available within polyply,
run:

```bash
polyply -list-lib
```

You should see that a library for martini3 polymers is implemented. Next, you can check
that the building block you require is present in the library by running:

```bash
polyply -list-blocks martini3
```

To check any other library simply replace `martini3` with the library name. Running the
command tells us that there are parameters for both PS and PEO present. Note that there are
not always links to combine different polymers with each other. Next, we want to invoke the
`gen_params` sub-program to generate an itp file. For a single polymer, the command has the
following basic structure. Run `polyply gen_params -h` for more options.

```bash
polyply gen_params -lib <library_name> -o <filename>.itp -name <name of polymer> -seq [<monomer-name>:<number of monomers>, ...]
```

To generate PEO of length 200 we simply need to provide the library name using `-lib`, a
name for the itp file using `-o`, a name using `-name`, and the sequence of the polymer
using `-seq`. In this case, the sequence is just 200 residues of EO (i.e. `-seq EO:200`).
The full command would be:

```bash
polyply gen_params -lib martini3 -o PEO200.itp -name PEO -seq EO:200
```

This generates the required itp file. To do the same for PS simply change EO to STYR. If you
need more complex linear polymer sequences, you can provide a text file with a list of
residues and provide it using the `-seqf` option. Branched polymers are also possible to
generate itp files for, but you have to provide a networkx `.json` describing the graph of
the branched polymer. This
[discussion](https://github.com/marrink-lab/polyply_1.0/discussions/296) offers an example.
The file also has to be provided using the `-seqf` option.

## 3 - Generating starting coordinates

To generate starting coordinates all you need is the target topology file. First, we need
the interaction parameters. Those you can download
[here](http://cgmartini.nl/index.php/martini-3-0). Subsequently, write the topology file.
In our case let's generate an amorphous system of 200 polymers. The topology file looks as
follows:

```
#include "martini_v3.0.0.itp"
#include "PEO200.itp"
[ system ]
; name
Melt of PEO 200

[ molecules ]
; name  number
PEO 200
```

The martini interaction itp file (i.e. `martini_v3.0.0.itp`) can be downloaded from
[here](https://github.com/marrink-lab/martini-forcefields/tree/main/martini_forcefields/regular/v3.0.0/gmx_files).
To create starting coordinates we call the `gen_coords` sub-program with the general command
looking as follows:

```bash
polyply gen_coords -p <topfilename> -o <filename>.gro -name <name> -dens <density>
```

In order to generate coordinates you have two options for setting the system size: (1) you
set the box size (i.e. `-box <x> <y> <z>`; keep in mind only rectangular boxes are currently
supported); (2) you provide the overall system density using `-dens`. Most of the time it is
convenient to provide the target system density. Note that sometimes it is convenient to set
the target density a little lower than what is expected in the simulation to gain a speed-up
in the structure generation. To further speed up the generation, you can install the
[numba package](https://numba.pydata.org/). For our example, we will generate PEO at a
target density of 1000 kg/m³, which is the Martini system density. The overall command for
our polymer amorphous system becomes:

```bash
polyply gen_coords -p system.top -o melt.gro -name PEO -dens 1000
```

This should take less than 60 seconds on a single CPU provided you installed numba.

!!! note
    Once you have generated your melt you will need to first run an energy minimization,
    followed by a short NpT equilibration. You can download the files for minimization and
    equilibration
    [here](https://github.com/marrink-lab/polyply_regression_tests/tree/main/tutorial_files/martini_melts).

The GROMACS commands are as follows:

```bash
gmx grompp -f min.mdp -c melt.gro -p system.top -o min.tpr
gmx mdrun -v -s min.tpr -deffnm min

gmx grompp -f pre_NpT.mdp -c min.gro -p system.top -o eq.tpr
gmx mdrun -v -s eq.tpr -deffnm eq
```

Note that to run the system as a polymer melt you need to increase the temperature above the
PEO melting temperature (~335 K).

## 4 - Generating starting coordinates for a mixed PEO/PS blend

To generate the polymer blend you simply have to generate an itp file for PS as well.
Subsequently, add it to your topology file and run the command above again. This will
generate the polymer blend. Edit the topology file to include PS:

```
#include "martini_v3.0.0.itp"
#include "PEO200.itp"
#include "PS200.itp"
[ system ]
; name
A blend of PEO and PS

[ molecules ]
; name  number
PEO 200
PS 200
```

Now run the coordinate generation again:

```bash
polyply gen_coords -p system.top -o blend.gro -name PEO_PS_blend -dens 1000
```

Again, energy minimization and equilibration are needed to get the system ready for
production.

## 5 - Generating starting coordinates for polymer solutions

If you already generated an itp file using the `gen_params` tool described in Section 2, a
polymer solution is generated as simply as the polymer blend. First, however, a solvent has
to be selected. Apart from water, Martini 3 offers a whole host of solvents. A comprehensive
list can be found in
[this file](https://github.com/marrink-lab/martini-forcefields/blob/main/martini_forcefields/regular/v3.0.0/gmx_files/martini_v3.0.0_solvents_v1.itp),
which you will also need to download and include in the topology. Additional, mostly
aromatic, solvents can be found
[here](https://github.com/marrink-lab/martini-forcefields/blob/main/martini_forcefields/regular/v3.0.0/gmx_files/martini_v3.0.0_small_molecules_v2.itp).

For example, PS is soluble in acetone but insoluble in water. To generate and compare both
systems, we can simply write the following two topology files and generate the coordinates
as before. Note that the concentration is adjusted by specifying an appropriate number of
solvent molecules. For this example, we use 8000 acetone molecules, a 1:4 ratio of monomers
to solvent. In subsequent commands we assume the file name to be `sol_sys.top`:

```
#include "martini_v3.0.0.itp"
#include "martini_v3.0.0_solvents_v1.itp"
#include "PS200.itp"
[ system ]
; name
A blend of PEO and PS

[ molecules ]
; name  number
PS 10
PPN 8000
```

The command for generating the system is pretty much the same as before. Note that the
Martini acetone density is about 970 kg/m³. The reason is that the mass of the Martini bead
is 72 amu, while a real acetone molecule has a mass of 58 amu. Computing the number density
by multiplying 970 kg/m³ with 58/72 gives 781 kg/m³ — the correct number density of acetone.
This example shows that you need to set the actual Martini solvent density, which is not
necessarily the same as the actual density. As Martini 3 reproduces densities quite well,
you can always compare the molecular weight of the Martini molecule with the real one and
scale the real density to get the Martini density.

```bash
polyply gen_coords -p sol_sys.top -o sol_sys.gro -name PS_PPN -dens 970
```

To generate the water system, just replace `PPN` with `W` and change the density to
1000 kg/m³. Minimization and equilibration work as before.

## 6 - Polymer liquid-liquid phase separation

Dextran and PEO form an LLPS when solvated in water above a critical concentration. Martini
can capture these phase effects. This particular system has already been
[published](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00757). To set up the system we
first have to generate the Dextran itp file. Dextran with a number-average molecular weight
of 65 is hardly branched and forms an LLPS with PEO. Thus we generate linear Dextran:

```bash
polyply gen_params -seq DEX:65 -o dextran.itp -name DEX -lib martini3
```

Subsequently, we need to build the system above the critical concentration. To speed up
phase separation and keep the system size small, we will choose a rather high concentration
in a small system. Note that for production the system sizes should not be too small. Write
the topology file called `llps.top`:

```
#include "martini_v3.0.0.itp"
#include "martini_v3.0_solvents.itp"
#include "dextran.itp"
#include "PEO.itp"

[ system ]
LLPS of dextran PEO in water

[ molecules ]
DEX 18
PEO 20
W   13693
```

Next, we generate the coordinates for the mixed system as done in the other examples. Only
in this case, we lower the maximum allowed force in the random-walk process by setting
`-mf 5000`. It is not strictly needed but helps with avoiding the occasional overlap.

```bash
polyply gen_coords -p llps.top -o llps.gro -name llps -dens 1000 -mf 5000
```

After energy minimization, you should see a largely mixed system of PEO and Dextran in
water. To facilitate phase separation you can change the barostat settings to
semi-isotropic. The phase separation takes about 500 ns before it becomes visible.

## 7 - Useful notes

- Run energy minimizations with the `-DFLEXIBLE` option to use flexible bonds.
- Energy minimizing using GROMACS double precision can help to relax close-contact
  configurations better.
- Run equilibration with the Berendsen barostat to relax to the target volume, or the
  C-rescale barostat for isotropic systems.
- The solvent density in Martini is not always the same as the real density, because Martini
  beads have a fixed mass. Compare the solvent mass to the real molecule mass to see if the
  density has to be scaled.
