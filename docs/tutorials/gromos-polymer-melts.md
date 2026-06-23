# Tutorial: GROMOS Polymer Melts

!!! warning
    This tutorial is under construction; use with caution. If you find any mistakes, unclear
    sections, or erroneous behaviour, please report it as an issue. Only with your feedback
    are we able to improve the tutorials and make them more helpful for everyone.

## 1 - Introduction

Polyply facilitates the generation of polymer melts. This tutorial illustrates how to set up
an atomistic melt for different homo-polymers and a statistical copolymer, and how to deal
with tacticity. In this tutorial we will consider poly(methyl acrylate) (PMA) and
poly(methyl methacrylate) (PMMA) using the GROMOS 2016H66 force-field [1].

## 2 - Generating a residue graph with appropriate chirality/tacticity

Both PMMA and PMA contain a chiral center in the monomeric repeat unit, which gives rise to
tacticity in those polymers. Tacticity essentially describes the relative chirality of
neighboring monomeric repeat units. In atactic polymers, the distribution is completely
random, whereas in isotactic polymers all monomeric repeat units have the same chirality. In
polyply the chirality has to be set as a node attribute in the residue graph. Internally the
labels R and S are used for vinyl polymers. However, it should be noted that these do not
necessarily correspond to the IUPAC definitions of R and S. They are rather used as relative
labels (i.e. two monomeric repeat units of S have the same chirality, while R and S have
opposite chirality). Thus a polymer of pure S or pure R would be considered isotactic,
whereas a statistical distribution would be atactic. To generate the sequence for any linear
polymer and add a label such as chirality to it, we use the `polyply gen_seq` tool as
follows:

```bash
polyply gen_seq -from_string <name:#monomers:1:resname-1.0> -o polymer.json -name polymer -seq name -label 0:chiral:R-0.5,S-0.5
```

In this command the `-from_string` flag tells the program we want to generate a polymer from
a string. The string consists of the name, followed by `:`, then the number of monomers, and
the degree of branching, which is 1 for a linear polymer. Note that you can provide several
blocks by copying the statement between the `<>`. Furthermore, the `<>` is not part of the
syntax; it simply indicates that this is one input string. The last field of the string
contains the residue names and probabilities for that residue in the polymer. For generating
atactic PMMA and PMA of 100 repeat units, the commands are:

```bash
polyply gen_seq -from_string A:100:1:PMA-1.0 -o PMA.json -name PMA -seq A -label 0:chiral:R-0.5,S-0.5
```

```bash
polyply gen_seq -from_string A:100:1:PMMA-1.0 -o PMMA.json -name PMMA -seq A -label 0:chiral:R-0.5,S-0.5
```

The `.json` files contain the graph information for atactic PMA and PMMA with labels
indicating the chirality for each residue. Note that running the command multiple times will
generate a different chirality distribution each time. For a single residue we just used
`resname-1.0`, but for having more than one residue the syntax is
`resnameA-probA,resnameB-probB`. So to generate the sequence for a statistical copolymer, we
simply have to adjust the string describing our A block and run the following command:

```bash
polyply gen_seq -from_string A:100:1:PMA-0.5,PMMA-0.5 -o PMMA_co_PMA.json -name PMMA_co_PMA -seq A -label 0:chiral:R-0.5,S-0.5
```

## 3 - Generating an itp file

Next we need to generate an itp file for the polymers that we want to study. Polyply is
shipped with a range of polymer input files collected in libraries, which can be used to
generate itp files for polymers. Note that you can also implement your own polymer
definition — to do so, check out the tutorial on
[writing .ff input files](../reference/writing-ff-input-files.md). To inspect which libraries
are available within polyply, run:

```bash
polyply -list-lib
```

You should see that a library for 2016H66 polymers is implemented. Next we want to invoke
the `gen_params` sub-program to generate an itp file. For a single polymer the command has
the following basic structure:

```bash
polyply gen_params -lib <library_name> -o <filename>.itp -name <name of polymer> -seqf <graph-file>.json
```

You see that we can specify a library with the `-lib` flag, and set the name of the itp file
with `-o`. The molecule name that appears in the `[moleculetype]` directive is given by
`-name`. Finally, we can provide the sequence of the polymer. To do that we use `-seqf` to
provide a `.json` input file that defines the sequence and can be generated as shown before.
Run `polyply gen_params -h` for more options. To generate the itp files for PMA, PMMA, and
the copolymer, we simply run the following commands:

```bash
polyply gen_params -lib 2016H66 -o PMA.itp -name PMA -seqf PMA.json
```

```bash
polyply gen_params -lib 2016H66 -o PMMA.itp -name PMMA -seqf PMMA.json
```

```bash
polyply gen_params -lib 2016H66 -o PMA_co_PMMA.itp -name PMA_co_PMMA -seqf PMMA_co_PMA.json
```

This generates all required itp files. If you open the itp file of PMMA and look for
`#ifdef eq_polyply`, you will see that within the itp there are some improper dihedral angles
defined using the GROMACS `#ifdef` syntax. Those dihedral angles are essential in producing
the proper chirality during the structure generation. On the other hand, if you open the PMA
itp file you will find that these dihedral angles are part of the force-field, because in the
case of PMA the chiral hydrogen is not modeled explicitly within GROMOS. Later we will check
that the chirality of all polymers is as expected.

## 4 - Generating starting coordinates

To generate starting coordinates all you need is the target topology file. First we need the
interaction parameters. Those you can download
[here](https://github.com/marrink-lab/polyply_regression_tests/tree/main/polyply_regression_tests/data/force_fields/2016H66.ff).
Next, write the topology file. Here we generate a melt of 200 polymers of all three species.
The topology file looks as follows:

```
#include "2016H66.ff/forcefield.itp"
#include "PMA.itp"
#include "PMMA.itp"
#include "PMA_co_PMMA.itp"
[ system ]
; name
Melt of PMA, PMMA, PMMA-co-PMA

[ molecules ]
; name  number
PMA 20
PMMA 20
PMA_co_PMMA 10
```

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
in the structure generation; subsequently, however, the system needs longer to relax. To
further speed up the generation you can install the
[numba package](https://numba.pydata.org/). For our example, we will generate the melt at a
target density of 1100 kg/m³, which is the density of PMMA at 413 K. The overall command for
our polymer melt becomes:

```bash
polyply gen_coords -p system.top -o melt.gro -name melt -dens 1100
```

This should take at most up to a minute on a single CPU provided you installed numba. Once
you have generated your melt you will need to first run an energy minimization. However,
make sure to set the `-Deq_polyply` flag when running an energy minimization and during
short equilibration. This will ensure that the chirality of PMMA is set correctly as
explained before. After energy minimization we recommend running a few nanoseconds with
flexible bonds using a 10 fs time-step in order for the system to relax further before using
constraints to possibly increase the time-step to 20 fs.

## 5 - Notes

- The `<>` are indications of a general syntax placeholder; don't actually use them in the
  command.
- In order to accommodate realistic polymer conformations the system has to be large enough.
  A good way to check this is by calculating the end-to-end distance expected for all/some
  polymers on the back of an envelope using the formula $EE = n \cdot l \cdot C_\infty$.
- The GROMOS family of force-fields uses the reaction-field cut-off method. In order to
  simulate the polymer melts adequately, the dielectric constant has to be set to a
  reasonable value for the bulk medium. PMMA has a dielectric constant of around 5, which we
  use in this example; SPC water on the other hand uses a dielectric constant of 60.
- Always run a short NVT relaxation simulation before the NpT relaxation and production run.

## 6 - References

[1] Horta, B.A., Merz, P.T., Fuchs, P.F., Dolenc, J., Riniker, S. and Hünenberger, P.H.,
2016. A GROMOS-compatible force field for small organic molecules in the condensed phase: The
2016H66 parameter set. *Journal of Chemical Theory and Computation*, 12(8), pp.3825-3850.
