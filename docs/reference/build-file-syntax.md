# Syntax: Build File

## Synopsis

The build file (extension `.bld`) allows users to steer certain aspects of the structure
generation and backmapping in polyply. At the moment polyply offers two types of options.
Build options are criteria that modify the acceptance of a step in the random-walk protocol in
addition to the overlap criterion. For example, there are geometric restraints, which assess
if a step within a region of space is valid or not. Another form of build option is a distance
restraint that checks if a step generates coordinates that obey specified distance restraints.
Besides build options, there are two more options that specify more global aspects of the
structure generation, namely the volume of a residue and the template coordinates.

!!! note
    In the syntax descriptions below, `<>` means "replace keyword". Do not copy the `<` or `>`
    signs.

## Template options

Polyply generates residue templates on the fly by a graph-embedding procedure. However, the
procedure becomes slow for large residues, or it can fail to generate correct templates,
especially if there are multiple chiral centers present. Even though a warning is emitted in
those cases, it is more convenient to provide templates used in the structure generation. This
is done by specifying a template using the `[ template ]` directive. Multiple templates can be
given by providing the template directive multiple times. The syntax is shown below.

```
[ template ]
resname <resname>
[ atoms ]
<atomname> <atomtype> <x> <y> <z>
[ bonds ]
<atomname> <atomname>
```

The atom names and atom types must exactly match those in the provided topology for the
residue. Furthermore, there can only be one template per residue. Coordinates should be given
in nm. If templates are provided and no volumes (see next section), the volume is estimated
from the residue template using the generic procedure.

## Volume options

As with the custom templates, it sometimes can be convenient to provide template volumes to
polyply instead of using the generic model. Volumes are defined as the sigma parameter of a
Lennard-Jones interaction function. Cross interactions between residues at the moment are
calculated from combination rules. The syntax is shown below.

```
[ volumes ]
<resname> <LJ_sigma_value>
```

Note that the unit of the LJ sigma value should be in nm. C6/C12 parameters are not supported
in this input. If volumes are provided they override any volumes estimated.

## Build options

Build options are grouped by molecule types and can be nested under each other. Therefore each
entry starts with the `[ molecule ]` directive, which allows you to select specific molecules
from the topology to apply the build options to. The syntax of the molecule directive consists
of the directive key and then, on a new line, the molecule name as listed in the topology and
the molecule indexes as defined by order of appearance in the topology file. Note molecules
start counting at 0. For example, to select the first 100 molecules in a PEO melt the molecule
directive would be written as shown below:

```
[ molecule ]
PEO 0 100
```

The build options are directives as well that can be listed under the `[ molecule ]` directive
(see table below). Each directive consists again of the directive key and then, on a new line,
a number of parameters describing the directive.

| Build option | Num. parms. | Order of parameters and units | Comment |
|---|---|---|---|
| `sphere` | 8 | resname; resid_start; resid_stop; in/out; x (nm); y (nm); z (nm); r (nm) | resids define a range of resids |
| `rectangle` | 10 | resname; resid_start; resid_stop; in/out; x (nm); y (nm); z (nm); a (nm); b (nm); c (nm) | resids define a range of resids |
| `cylinder` | 9 | resname; resid_start; resid_stop; in/out; x (nm); y (nm); z (nm); r (nm); dz (nm) | resids define a range of resids |
| `persistence_length` | 4 | WCM; lp (nm); first resid; last resid | resid define the first and last residue |
| `distance_restraints` | 3 | resid 1; resid 2; distance (nm) | resid define two different residues |
| `rw_restriction` | 4 | resname; resid_start; resid_stop; nx (nm); ny (nm); nz (nm); angle (deg) | n refers to normal |

For example, to place the first 10 polymer molecules with name `polym` and 100 residues of
randomly placed PS or PMA into a sphere of radius 4 located at a central point with coordinates
`x=10.0, y=10.0, z=10.0`, the directive would be:

```
[ molecule ]
polym 0 10
[ sphere ]
PS 0 100 in 10.0 10.0 10.0 4.0
PMA 0 100 in 10.0 10.0 10.0 4.0
```

Note that here, because we do not want to specify every single residue along the chain, we
simply set the resid range from 0 to 100. Polyply will print some warnings but correctly
identify the residues we want.

### Resid intervals

Most directives require a residue name and a range of resids to be specified. This gives users
the option to specifically target only certain residues within a molecule. Note that resids
always start counting at 1, and the range is taken over the interval (start, stop]. Except for
distance restraints and persistence length, all residue specifications are only considered if
the combination of molecule index, molecule name, residue name, and resid all match. Otherwise
they are skipped and a warning is issued. However, this allows very convenient writing of
entries. For example, in a protein of 100 residues one wants to select only the ALA residues.
Then using `ALA 0 100` will result in only the alanines being selected and all other residues
being skipped, instead of having to single out every alanine residue.

### Geometrical restraints

Geometrical restraints check if a new point is within a certain volume in space and then
either cause the algorithm to accept or reject the step. Regions in space can be defined as
spheres, rectangles, or cylinders, plus one of the keywords `in` or `out` to specify if the
step needs to be inside or outside the volume. The regions in space are then defined by the
following rules:

- **a sphere** is defined by its center and a radius.
- **a rectangle** is defined by its center and three distances from the center to the faces.
- **a cylinder** is defined by its center, a radius, and the distance to the cylinder wall.

Thus each directive consists of the x, y, z coordinates of the central point followed by the
additional parameters defining the shape.

### Persistence length

The persistence length directive contains the entries specifying how to sample the persistence
length. It is important to be aware that the persistence length is taken into account by
generating an end-to-end distance distribution and then sub-sampling that distribution to
generate the end-to-end distances. However, when only a single polymer molecule is specified,
the algorithm will take **one** end-to-end distance from the distribution. In addition, for
very stiff molecules, if the polymer is too long the end-to-end distance is not a good
descriptor anymore, and instead the persistence length should be applied on segments of the
chain.

The directive itself consists of one line. First the keyword WCM is given, which indicates
that the worm-like-chain model is used. Currently only the WCM model is implemented. Following
the WCM keyword the persistence length is given in nm. Afterwards two integer resids have to be
given, indicating the first and the last residue of the chain. Using this syntax, also branched
or grafted polymers can be treated by selecting the appropriate start and end point of the
chain. Note that there has to be one unique shortest path connecting the two residues.

Note that you can add multiple persistence length directives for a molecule, for example if you
have a block copolymer.

### Distance restraints

Distance restraints consist of two resids followed by a distance at which they need to be
restrained. Note that distance restraints are by default satisfied up to +/- 1 step length.
Multiple distance restraints, and also overlapping restraints, can be specified. However, for
multiple restraints it does not mean the algorithm will find a solution. We perform no check as
to whether the input is valid and can actually be satisfied. As always, trying multiple times
is a good course of action in case the restraints cannot be satisfied.

### rw_restriction

The `rw_restriction` directive allows you to restrict the direction of the random-walk along a
specific vector. It works by specifying that vector and an angle. The vector describes a plane,
and the angle describes the maximum allowed angle of the new step with that plane. If the angle
is 90 degrees it means the random-walk will not cross "back" across the plane, and if it is even
lower it means it will approximate a step in the direction of the plane normal. This option can
be used to generate brushes, as shown in the
[PEGylated lipid tutorial](../tutorials/pegylated-lipid-bilayers.md).
