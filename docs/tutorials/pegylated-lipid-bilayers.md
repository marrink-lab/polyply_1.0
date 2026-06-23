# Tutorial: PEGylated Lipid Bilayers

## 1 - Introduction

With polyply it becomes easy to combine your polymer with any existing bio-molecules. One
such use case is the creation of PEGylated lipid bilayers. PEGylated lipids are lipids to
which PEG is covalently attached. Depending on the grafting density (i.e. concentration of
those lipids in the bilayer) the chain dimensions fall into one of two regimes. In the low
concentration regime, known as the "mushroom regime", the chains are well-separated, interact
minimally, and therefore move freely within a space approximating a half-sphere [1]. In
contrast, in the high grafting density regime ("brush regime"), the polymer chains are close
in space and repel each other. This repulsion leads to more extended chain dimensions than in
the mushroom regime. More information about PEGylated lipid bilayers can be found here [2,3].
In this tutorial, we learn how to create PEGylated bilayers in the two regimes using the
Martini force field. Please make sure to consult the [Useful Notes](#4-useful-notes) section
before starting your own project. When using the Martini 3 force field it is essential to
adjust the PEG–phosphate interaction parameters as described.

## 2 - Generating an itp file

First, we need to generate an itp file for the PEGylated lipid we want to study. To do so we
have to
[download](https://github.com/marrink-lab/martini-forcefields/tree/main/martini_forcefields/regular/v3.0.0/gmx_files)
the Martini 3 [4] force-field files, which also contain the lipid definitions
(`martini300/martini_v3.0.0_phospholipids_v1.itp`). Note that connecting PEG to lipids is
currently only supported for 3 lipids (DPPE, POPE, DOPE). Once we have the parameter files we
simply invoke the itp file generation tool (`polyply gen_params`) as follows:

```bash
polyply gen_params -f martini_v3.0.0_phospholipids_v1.itp -lib martini3 -o pegylated_lipid.itp -name PEL -seq <lipid-name>:1 PEO:<number of monomers>
```

For this command, we have to provide the lipid itp file using the `-f` flag. Next, we have to
tell polyply we want to use the library of Martini 3 using `-lib martini3`. Besides the name
of the output file provided with the `-o` flag and the name of the molecule set by `-name`,
we also need to define the macromolecule residue graph, which in this case is a linear
sequence. We want one lipid molecule (i.e. `<lipid-name>:1`) to be connected to several
monomers of PEO (i.e. `PEO:<number of monomers>`). The lipid name is the molecule name
specified in the lipid itp file.

Let us generate a specific example, namely PEGylated POPE with 50 residues of PEG terminated
with an OH group. We will use this lipid as an example in the rest of the tutorial. To
generate the itp file, run the following command:

```bash
polyply gen_params -f martini_v3.0.0_phospholipids_v1.itp -lib martini3 -o PEL.itp -name PEL -seq POPE:1 PEO:50 OHter:1
```

## 3 - Generating starting coordinates

Now that we have the itp file, we will learn how to generate the starting coordinates. This
is done in two steps: (1) first use
[insane](http://cgmartini.nl/index.php/tools2/proteins-and-bilayers) or
[TS2CG](https://github.com/marrink-lab/TS2CG) to generate a lipid bilayer of your choice.
Don't add water or ions yet! (2) Subsequently use the polyply coordinate generation tool to
add the polymer. For the sake of this tutorial we generate a lipid bilayer of 100 POPC lipids
in each leaflet. If you are using insane this can be done by running the following command:

```bash
python2 insane.py -l POPC:95 -l POPE:5 -o bilayer.gro -dz 15.0 -x 8.0 -y 8.0
```

Now that we have the bilayer, let's begin by generating coordinates for the mushroom state.

### 3.1 - Coordinates for the mushroom state

The first thing we need is a topology file. Let's call it `system.top`. This file needs to
describe the complete system that we want to generate. The topology file should then look
like this:

```
#include "martini_v3.0.0.itp"
#include "martini_v3.0.0_ions_v1.itp"
#include "martini_v3.0.0_solvents_v1.itp"
#include "martini_v3.0.0_phospholipids_v1.itp"
#include "PEL.itp"
[ system ]
; name
PEGylated lipid mushroom

[ molecules ]
; name  number
PEL          5
POPC         95
PEL          5
POPC         95
```

The mushroom state is characterized by an unrestricted chain tethered to the lipid bilayer.
So there is no specific directional restriction to the growing of the polymer chain. Thus we
can use the following command to "grow" the polymer on top of the lipid bilayer:

```bash
polyply gen_coords -p system.top -o struct.gro -res PEO OHter -c bilayer.gro -name test -box 8.0 8.0 20.0 -split POPE:HEAD-NH3:TAIL-PO4,GL1,GL2,C1A,D2A,C3A,C4A,C1B,C2B,C3B,C4B
```

To generate the structure we always need to specify the initial coordinates of the lipid
bilayer (i.e. `-c bilayer.gro`), the residues that need to be built (`-res PEO OHter`), and
the box (`-box 8.0 8.0 20`). In addition, we choose to split the residue POPE into two
separate residues describing the lipid head group and the lipid tail region. Run
`polyply gen_coords -h` to get more information on the splitting syntax. Splitting the
residues is needed because polyply generates a super-coarse-grained 1-bead-per-residue model.
However, PEO is attached to the head group and we want to start "growing" the chain from
there. More information can be found in the
[publication](https://arxiv.org/abs/2105.05890). When you change the lipid type you might
have to adapt the names. However, the head group always contains an NH3 and PO4 bead.
Finally, you can add water to the system with your favorite tool for solvation, or skip ahead
to use polyply.

### 3.2 - Coordinates for the brush regime

In the brush regime the grafting density is higher, the chains tend to be more elongated and
more tightly packed. When we generate the initial structure we need to take those factors
into account. This can be done very easily by creating a build file, which can be used to
specify more complex building options. First, we need to increase the grafting density. We
generate a new lipid bilayer:

```bash
python2 insane.py -l POPC:75 -l POPE:25 -o bilayer_brush.gro -dz 33.0 -x 8.0 -y 8.0
```

Then we change the number of PEL molecules in the topology file to 25 for each leaflet. Your
topology file should look like this now:

```
#include "martini_v3.0.0.itp"
#include "martini_v3.0.0_ions_v1.itp"
#include "martini_v3.0.0_solvents_v1.itp"
#include "martini_v3.0.0_phospholipids_v1.itp"
#include "PEL.itp"
[ system ]
; name
PEGylated lipid brush

[ molecules ]
; name  number
PEL          25
POPC         75
PEL          25
POPC         75
```

Next create a build file, which looks as follows:

```
[ molecule ]
; molname
PEL  0  25
[ rw_restriction ]
; resname start stop  normal   angle
 PEO    3   52    0 0 1     90.0

[ molecule ]
; molname
PEL  100 125
[ rw_restriction ]
; resname start stop  normal   angle
 PEO    2   51    0 0 -1     90.0
```

In this build file, we can specify building options for specific molecules in the system.
Which molecules we want to target is specified by the `[ molecule ]` directive that is
followed by a line of the following format:
`<molecule_name> <starting molecule index> <ending molecule index>`. Note molecule indices
are counted as total, not per species. For this example, we want to specify options for the
first 25 lipids and the lipids 100 to 125. In this specific case, the first 25 are in the
upper leaflet and the others are in the lower leaflet. Next, we write the `[rw_restriction]`
directive. This directive can be used to specify how we want to restrict the random walk used
to grow the polymer chains. It has the following syntax:
`<resname> <starting resid> <last resid> <x> <y> <z> <angle in degree>`. Each line specifies
which residue of the molecule we want to target by residue name and the range of residue IDs,
followed by a plane normal vector and an angle. Using this syntax the random walk will be
restricted by the following rule: *A step is only allowed if the angle between the step vector
and a vector specified in the `[rw_restriction]` directive is less than the angle specified in
that same directive.* In our case we want the chains in the upper leaflet to be stretched in
the positive z-direction and in the lower leaflet in the negative z-direction. Thus we specify
the positive and negative z-axis as our reference vectors. The angle is set to 90 degrees,
which effectively means a step can never go back in the z-direction but is free to move in x
and y. Decreasing the angle will give even straighter chains. Using this build file the
PEGylated bilayer brush is simply generated by running:

```bash
polyply gen_coords -p system.top -o struct.gro -res PEO OHter -c bilayer_brush.gro -name test -box 8.0 8.0 33.0 -split POPE:HEAD-NH3:TAIL-PO4,GL1,GL2,C1A,D2A,C3A,C4A,C1B,C2B,C3B,C4B -b build_file.bld
```

Running this command will yield the desired polymer brush. Again, after solvating the system
and running an energy minimization, the procedure is complete.

### 3.3 - Adding water and ions

In principle, polyply can be used to add water as well as ions to structures. The only
downside is that you will have to more or less accurately guess the number of water molecules
and ions. However, it also has the advantage that you can exclude regions in space where
water should not be added. For example, `gmx solvate` might add water to the lipid bilayer
tail region, which can perturb the bilayer. Let us try water placement for the brush we just
created. First, we need to add the water to the topology file. Add the following line to the
topology file from the last step: `W 12412`. Next, we add the following entry to the build
file:

```
[ molecule ]
; molname
W  200 14412
[ rectangle ]
; resname start stop  inside-out  x  y  z    a  b  c
W  0  1   out               4.0 4.0 16.5  4.1 4.1 1.5
```

Now we invoke the rectangle directive, which looks very similar to the previous directive.
However, in this case the syntax is:
`<resname> <starting resid> <last resid> <in or out> <x> <y> <z> <a> <b> <c>`, where we
specify a point and three distances along (a, b, c) from a central point to the faces of a
rectangle. Together with the keyword `out` it means all water molecules need to be outside a
rectangle defined by this syntax. After having edited all files, run the following command:

```bash
polyply gen_coords -p system.top -o struct.gro -res PEO OHter -c bilayer_brush.gro -name test -box 8.0 8.0 33.0 -split POPE:HEAD-NH3:TAIL-PO4,GL1,GL2,C1A,D2A,C3A,C4A,C1B,C2B,C3B,C4B -b build_file.bld
```

After energy minimization, your system is ready to go. Note that PEGylated lipids are
charged, so you need to add ions. It is possible to add ions with polyply, but note that in
the Martini 3 itp files you need to give all ions different residue names first.

## 4 - Useful Notes

- Implemented lipids: DPPE, POPE, DOPE.
- If you'd like another PEGylated lipid, please raise an issue and we'll help you asap. But
  also make sure to give issue
  [#167](https://github.com/marrink-lab/polyply_1.0/issues/167) a quick look.
- In order to add PEGylated lipids to a vesicle, please have a look at
  [this tutorial](http://cgmartini.nl/index.php/2021-martini-online-workshop/tutorials/559-8-polyply).
- Always run an energy minimization before running an NpT relaxation.
- To see other libraries implemented, run `polyply -list-lib`.
- This tutorial is affected by issue #530; thus resids for the lipid start at 0.
- **Currently, the Martini 3 PEG interacts too strongly with the bilayer. To get better
  behavior, change the Q5 (i.e. phosphate)–SN3r interaction to 3.0 kJ/mol. This change
  reproduces bilayer affinities compared to all-atom simulations and resolves the problem. If
  you use these improved parameters please cite
  [10.33612/diss.756637591](https://doi.org/10.33612/diss.756637591). For a detailed
  discussion please consult Chapter 5, Section 3.5 of
  [10.33612/diss.756637591](https://doi.org/10.33612/diss.756637591). Otherwise, the Martini 2
  PEG parameters reproduce experimental trends in the bilayer–PEG interactions better.**
- The PEG Martini 2 parameters are not part of the default `martini.itp` bead definitions.
  The special parameters including PEG can be found
  [here](http://cgmartini.nl/index.php/force-field-parameters/polymers2/442-polymers.html?dir=Linear&lipid=PEG).

## 5 - References

[1] Grünewald, F., Rossi, G., de Vries, A.H., Marrink, S.J. and Monticelli, L., 2018.
Transferable MARTINI model of poly(ethylene oxide). *The Journal of Physical Chemistry B*,
122(29), pp.7436-7449.

[2] Lee, H. and Larson, R.G., 2016. Adsorption of plasma proteins onto PEGylated lipid
bilayers: the effect of PEG size and grafting density. *Biomacromolecules*, 17(5),
pp.1757-1765.

[3] Lee, H. and Pastor, R.W., 2011. Coarse-grained model for PEGylated lipids: effect of
PEGylation on the size and shape of self-assembled structures. *The Journal of Physical
Chemistry B*, 115(24), pp.7830-7837.

[4] Souza, P.C., Alessandri, R., Barnoud, J., Thallmair, S., Faustino, I., Grünewald, F.,
Patmanidis, I., Abdizadeh, H., Bruininks, B.M., Wassenaar, T.A. and Kroon, P.C., 2021. Martini
3: a general purpose force field for coarse-grained molecular dynamics. *Nature Methods*,
18(4), pp.382-388.

[5] Grünewald, Fabian. "Material design using Martini: Accelerating discovery through
coarse-grained simulations." (2023).
