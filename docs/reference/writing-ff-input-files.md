# Writing .ff Input Files

## 1 - Introduction

In this tutorial we will learn how to write vermouth (`.ff`) input files, which are used by
polyply to generate polymer itp-files. This tutorial covers standard cases and concepts. As a
hands-on example we will generate input files for atomistic polyethylene glycol in the GROMOS
2016H66 force-field.

Polyply is a residue-oriented program. This means it is based upon the fact that polymers can
be split into residues (i.e. monomers), and those residues are then to be connected. The input
files reflect this notion in the sense that there exist two simple directives. Here we call a
directive anything that has the following format: `[ directive ]`. If you are familiar with
GROMACS itp files, you will know that to define a molecule you need to start by writing a
`[moleculetype]` directive followed by an `[atoms]` directive, a `[bonds]` directive, and so
forth. The vermouth input file format is very similar to this style of writing input files.

Coming back to polymers: we need in general two directive types. The `[moleculetype]` directive
is used to define all atoms and bonded interactions that are within a single residue. On the
other hand, the `[link]` directive specifies all bonded interactions that link two residues
together. Both of them together will enable you to write custom polymer itp-files in a very
efficient manner. First we see how to write the `[moleculetype]` directive.

!!! note
    Make sure to also read the [notes](#6-useful-notes) at the end of the tutorial, as they
    detail some exceptions and gotchas.

## 2 - Definitions of residues

Residues (i.e. monomers in a polymer) are defined by writing the `[moleculetype]` directive. It
follows the general format of a GROMACS molecule itp file. So writing an input file for a
residue is like writing the itp-file for that residue. The following itp file is the PEG repeat
unit file for atomistic PEO. Note that the repeat unit here is taken to be -[CH2-O-CH2]-.

```
[ moleculetype ]
PEO 3
[ atoms ]
1 CH3 1 PEO EC1 1  0.29 14.0
2 OE  1 PEO O1  2 -0.58 16.0
3 CH3 1 PEO EC2 3  0.29 14.0
[ bonds ]
1 2 2 gb_18
3 2 2 gb_18
[ angles ]
3 2 1 2 ga_15
```

There are a few important things to note:

1. All atom names need to be unique within one residue.
2. Residue name and molecule name should be the same.
3. Atom indices in the interactions need to match those in the atoms section.
4. Parameters such as the force constant are taken as is; their number is not verified.

The last point gives you the freedom to write, for example, the bond between atom 1 and atom 2
as `1 2 2 gb_18` or alternatively as `1 2 2 1.4300000e-01  8.1800000e+06`, which is the GROMOS
definition for that bond. You could also write it as a blank (i.e. `1 2 2`).

Another thing to note about this PEG residue is that we have written atom 1 and 3 as `CH3`,
meaning the itp file is equivalent to dimethyl ether. This is important because if we call
polyply to ask for a single residue, we will automatically get it as CH3-terminated. Later,
when writing the links, we will account for the fact that the atom type in the polymer should
be `CH2`. In the case of all-atom force-fields it is frequently necessary to add hydrogen atoms
or CH3 groups to the terminal residue. In this case you will have to define these as a separate
residue and make sure to include it in the sequence. See the definitions of
[AA P3HT](https://github.com/marrink-lab/polyply_1.0/blob/master/polyply/data/gromos53A6/Hter.ff)
as an example.

For the residue definition we are already done, so we move on to the links.

## 3 - Definitions of links

Next we will have to define a number of links which specify how we string together the PEO
residues. The link definition begins by writing the link directive as follows:

```
[ link ]
resname PEO
```

Note that if your link can apply to multiple residues you can write
`resname "<resname1>|<resname2>|<resname3>"` etc., but in this case we only concern ourselves
with PEO. In general it is recommended to write every bonded directive into its own link
definition. That is, writing a separate link directive for all the bonds, angles, and
dihedrals. For PEO we will need bonds, angles, dihedrals, and pairs.

### 3.1 - Definition of bonds

Let us start with the bond. When two PEOs connect we expect there to be a bond between the
atoms `EC1` and `EC2` of the next residue. So we write the following link entry:

```
[ link ]
resname "PEO"
[ atoms ]
EC2 { }
+EC1 { }
[ bonds ]
+EC1 EC2 2 gb_27
```

In this link entry we note the atom names involved in the interaction in the atoms directive,
followed by empty curly braces. Then we define the bonds directive in the same way as in an itp
file; however, the atom names (e.g. EC2) are used instead of the numbers. In addition, the
relative residue ID needs to be defined. This link is supposed to apply between the atom EC2 of
one residue and the atom EC1, which belongs to the next residue. The next residue has a resid
+1 higher, so we use the prefix `+`. However, we still have the atom types CH3. To take care of
that we add the keyword `replace` in the atoms directive as follows:

```
[ link ]
resname "PEO"
[ atoms ]
EC2 {"replace": {"atype": "CH2"}}
+EC1 {"replace": {"atype": "CH2"}}
[ bonds ]
+EC1 EC2 2 gb_27
```

The `replace` directive will replace the atom types of EC2 and +EC1 whenever this link is
applied. In this way, when we get a bond between two residues, we also change the atom types.

### 3.2 - Definition of angles

Next we define the links for angles in the same way as for bonds. Nothing new about those. Note
that we can omit the atoms directive in this case, because we don't use it.

```
[ link ]
resname "PEO"
[ angles ]
EC2 +EC1 +O1 2 ga_15
[ link ]
resname "PEO"
[ angles ]
+EC1 EC2 O1 2 ga_15
```

### 3.3 - Definition of dihedrals

Finally, PEG also requires dihedrals. In GROMOS there are three sets of three dihedrals defined
for the polyether bonds. Each set of dihedrals describes one torsion and becomes its own link
as shown below.

```
[ link ]
resname "PEO"
[ dihedrals ]
EC1 O1 EC2 +EC1 1 gd_42 {"version":"1"}
EC1 O1 EC2 +EC1 1 gd_43 {"version":"2"}
EC1 O1 EC2 +EC1 1 gd_44 {"version":"3"}

[ link ]
resname "PEO"
[ dihedrals ]
O1 EC2 +EC1 +O1 1 gd_45 {"version":"1"}
O1 EC2 +EC1 +O1 1 gd_46 {"version":"2"}
O1 EC2 +EC1 +O1 1 gd_47 {"version":"3"}

[ link ]
resname "PEO"
[ dihedrals ]
EC2 +EC1 +O1 +EC2 1 gd_42 {"version":"1"}
EC2 +EC1 +O1 +EC2 1 gd_43 {"version":"2"}
EC2 +EC1 +O1 +EC2 1 gd_44 {"version":"3"}
```

In general, interactions overwrite each other. For example, taking the last set of dihedrals we
would usually only get gd_44, because the other two dihedral interactions are overwritten. There
can only be one interaction of the same type for each set of indices. To circumvent this problem
the `version` keyword can be used between curly braces. This way we tell polyply not to overwrite
those dihedrals but to keep all three of them. This is a general feature and can be used for any
link directive.

### 3.4 - Definition of pairs

Last but not least we need to introduce pair interactions. Pairs are almost the same as the
interactions we have seen before. However, they have one more directive called `[edges]`. You
can think of edges like bonds between atoms. So why do we need to add them here? The reason is
that there are no clear rules for how to connect pairs.

```
[ link ]
resname "PEO"
[ atoms ]
EC1 { }
O1 { }
EC2 { }
+EC1 { }
+O1 { }
+EC2 { }
[ pairs ]
EC1 +EC1 1
O1 +O1 1
EC2 +EC2 1
[ edges ]
EC1 O1
O1 EC2
EC2 +EC1
+EC1 +O1
+O1 +EC2
```

## 4 - Virtual sites

Virtual sites are not present in PEG; however, many coarse-grained molecules use these types of
special particles. There are a number of virtual-sites available in GROMACS. The GROMACS syntax
for all of them is to first write the atom that is to be constructed as the virtual-site,
followed by the type and the constituting atoms and additional parameters if needed. This format
interferes with the vermouth parser that underlies polyply. Therefore virtual-sites require a
special syntax. Instead of writing it the GROMACS way, virtual sites are written as the virtual
site atom, followed by the constituting atoms, the `--`, and the parameters. If the virtual site
is called V and is formed by linear combination of the atoms A, B, C, the ff-directive looks as
follows:

```
; polyply vsn definition
[ virtual_sitesn ]
V A B C -- 1
```

## 5 - Making annotations with `{}`

There are some handy features that enrich the vermouth ff-file syntax to allow you to group
certain interactions or add comments. Both features help debugging wrong input and cleaning up
your parameter files. In general one can add after each interaction line the statements
`{comment: your_comment}` and/or `{group: your_group_name}`. The comment syntax will simply add
a comment after the interaction, whereas the group syntax actually groups interactions of the
same group names.

!!! warning
    This syntax does not work for virtual-sites and exclusions due to a bug in the parser.
    Instead you need to use the following approach for virtual-sites and exclusions:

```
[ exclusions ]
A B -- {"group": "group_name", "comment": "this is an exclusion"}
[ virtual_sitesn ]
V atom1 atom2 atom3 atom4 -- 1 {"group": "group_name", "comment": "this is a VSN virtual-site"}
```

## 6 - Useful notes

- Make sure to always verify, based on a small fragment, that all interactions are present.
- Virtual-sites, exclusions, and pairs must have an `[edges]` directive.
- If you have multiple dihedral or angle entries for the same atoms, make sure to use the
  `version` syntax.
- The order operator `+` always means +1 in the resid. If your resids have a larger gap you need
  to use the `>` operator instead.
- Exclusions and virtual sites require a special syntax also in combination with the annotations
  — see section 5.
