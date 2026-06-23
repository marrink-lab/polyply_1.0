# Tutorial: Martini 3 IDPs / Proteins

## Preliminaries

We can use the Martini 3 library in polyply to generate topologies for disordered proteins
from a sequence fasta file. These topologies have been adjusted from the default Martini 3
amino acid topologies to adjust protein–water interactions and improve the bonded parameters
for IDPs.

!!! warning
    This is **not** designed to generate topologies for Martini 3 proteins with folded
    domains. If you are not sure you are working with an IDP, check your structure and
    sequence before using this tool. [Metapredict](https://metapredict.net/) may be a useful
    tool for checking the sequence in particular. If you have a multidomain protein, it is
    best to use Martinize2 as described in the
    [Martini 3 Go model paper](https://www.nature.com/articles/s41467-025-58719-0). In any
    significant disordered region, Martinize2 may be used to selectively apply disordered
    parameters to the correct regions. For more detail on how disordered domains can be
    handled in Martinize2, please see the
    [documentation](https://vermouth-martinize.readthedocs.io/en/latest/tutorials/water_biasing.html).

Please cite the [preprint](https://www.nature.com/articles/s41467-025-58719-0) that describes
this work.

## Parameter generation

To begin, we need a fasta file with the IDP sequence. The fasta file must specify `PROTEIN`
in the header for polyply to interpret it correctly. Here, we use the example of an artificial
disordered protein, as designed by
[Dzuricky et al.](http://www.nature.com/articles/s41557-020-0511-7), with 10 octapeptide
repeat units. We'll call the file `WT10.fasta`:

```
> WT10 PROTEIN
SKGPGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGY
```

Once we have our disordered sequence, we can use the `gen_params` program of polyply to
generate the simulation input topology. Polyply has two force fields available with which to
model IDPs: Martini3-IDP and GōMartini with water biasing. The two force fields are specified
by `-lib martini3` and `-lib martini3-go` respectively. We will cover both methods here.

```bash
polyply gen_params -seqf WT10.fasta -name WT10 -o WT10.itp -lib martini3
```

This will generate a topology file containing the parameters for the input protein.

### Using water biasing

The GōMartini 3 approach to IDPs uses virtual Go sites along the backbone to effectively
adjust the backbone–water interaction. Using the protocol above to generate parameters
automatically introduces these virtual sites (along with other improved bonded interactions).
For example, the first few residues of the topology for the WT10 IDP discussed above now read:

```
[ atoms ]
  1 Q5    1 SER BB   1    1
  2 TP1   1 SER SC1  1  0.0
  3 VS    1 SER CA   1  0.0 0.0
  4 P2    2 LYS BB   2  0.0
  5 SC3   2 LYS SC1  2  0.0
  6 SQ4p  2 LYS SC2  2  1.0
  7 VS    2 LYS CA   2  0.0 0.0
  8 SP1   3 GLY BB   3  0.0
  9 VS    3 GLY CA   3  0.0 0.0
 10 SP2a  4 PRO BB   4  0.0
 11 SC3   4 PRO SC1  4  0.0
 12 VS    4 PRO CA   4  0.0 0.0
...
```

where an atom called `CA` of type `VS` has been introduced into each residue. Before using
the input files generated with this method you must ensure that, in your force field
definition (i.e. `martini_v3.0.0.itp`) file:

1. `VS` is defined in your `[ atomtypes ]` directive, e.g.:

```
...
[ atomtypes ]
...
TX1er 36.0 0.000 A 0.0 0.0
W  72.0 0.000 A 0.0 0.0
SW 54.0 0.000 A 0.0 0.0
TW 36.0 0.000 A 0.0 0.0
U  24.0 0.000 A 0.0 0.0
VS 0.00 0.000 V 0.0 0.0

[ nonbond_params ]
    P6    P6  1 4.700000e-01    4.990000e+00
    P6    P5  1 4.700000e-01    4.730000e+00
    P6    P4  1 4.700000e-01    4.480000e+00
...
```

2. An interaction is defined between `VS` and `W` in your `[ nonbond_params ]` directive,
   e.g.:

```
...
 TX2er  SQ1n  1 3.660000e-01    3.528000e+00
 TX2er  TQ1n  1 3.520000e-01    5.158000e+00
 TX1er   Q1n  1 3.950000e-01    1.981000e+00
 TX1er  SQ1n  1 3.780000e-01    3.098000e+00
 TX1er  TQ1n  1 3.660000e-01    4.422000e+00
    VS    W   1 0.4650000000    0.5000000000
```

The suggested parameters for the latter from the GōMartini 3 paper are $\sigma = 0.465$ and
$\epsilon = 0.5$, representing an increase in the strength of the protein–water interaction of
around 10%. However, if you find your IDP does not perform well with these parameters, then
the value of $\epsilon$ can be readily adjusted.

Once these additional parameters have been included in the input force field files, the IDP
topologies can be used as with any other input files for preparing simulations with polyply or
running them with GROMACS.

## Coordinate generation

Coordinates for (ensembles of) IDPs can be generated using the `gen_coords` program of
polyply, as described in the [Quick Start](../quick-start.md). One common use of polyply is to
set up pre-phase-separated systems, which can be achieved using a
[build file](../reference/build-file-syntax.md). For the example above, the following build
file could be used to build a slab in the middle of a rectangular (10 x 10 x 30 nm) box:

```
[molecule]
WT10 0 100
[rectangle]
; resname; resid_start; resid_stop; in/out; x (nm); y (nm); z (nm); a (nm); b (nm); c (nm)
SER 0 90 in 15 15 15 5 5 5
LYS 0 90 in 15 15 15 5 5 5
GLY 0 90 in 15 15 15 5 5 5
PRO 0 90 in 15 15 15 5 5 5
ARG 0 90 in 15 15 15 5 5 5
ASP 0 90 in 15 15 15 5 5 5
GLN 0 90 in 15 15 15 5 5 5
TYR 0 90 in 15 15 15 5 5 5
```

With a topology file that looks like:

```
#include "martini_v3.0.0.itp"
#include "WT10.itp"

[ system ]
my system

[ molecules ]
WT10 100
```

and the following command:

```bash
polyply gen_coords -p topol.top -box 10 10 30 -b build.bld -o newbox.gro
```

Note that this will generate warnings about residues not found. For this purpose, the warnings
can be ignored.

## Protein modifications

As of polyply v1.7.0, `polyply gen_params` supports modifications of proteins. Modification
syntax is `<resname><resid>:<target>`. For example:

```bash
polyply gen_params -lib martini3 -seq GLY:10 -name pGLY -o pGLY.itp -mods GLY1:N-ter GLY10:C-ter
```

will generate the parameters for polyglycine with 10 residues, with N and C termini at neutral
pH. Note that these terminal modifications are applied automatically when polyply determines
that the input sequence is a protein, so the same topology would be achieved with:

```bash
polyply gen_params -lib martini3 -seq GLY:10 -name pGLY -o pGLY.itp
```

In addition to terminal modifications, many of the usual protein modifications available in
Martinize2 are available, and may be combined however is desired. For example:

```bash
polyply gen_params -lib martini3 -seq HIS:5 -name HIS5_mods -o HIS5_mods.itp -mods HIS1:HIS-HD HIS1:NH2-ter
```

generates a histidine pentapeptide with a neutralised N terminal, and with the same histidine
side chain mutated to be representative of neutral histidine with hydrogen on the delta carbon.
