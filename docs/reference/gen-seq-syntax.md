# Syntax: gen_seq

Here we briefly describe the general usage and logic behind the program. The command-line
input to `gen_seq` are strings, which define macros. Macros are graphs and represent fragments
of residue graphs. Multiple macros can then be combined to form the residue graph of interest.
For example, one macro can describe a linear fragment and another one a branched fragment.
Combining them gives a residue graph describing a Janus-like particle. Macros are defined as
strings, with the following syntax:

```
<macro_name>:<#blocks>:<#branches>:<residues>
```

The `<macro_name>` field is an arbitrary name of that macro, the `<#blocks>` field should be
the number of residues for linear graphs and the number of generations for branched polymers.
Consequently, `<#branches>` is the degree of branching of that polymer. Only regular branching
is supported. The `<residues>` field can be used to define what the residue composition is
within that macro using the following syntax:

```
<residue_name-probability, other_residue_name-probability>
```

The residue name can be any string and the probability is the probability of having that
residue anywhere in the chain that is described by the macro. Let us consider some examples for
making macros. To generate a linear chain of 10 monomers of PEO one would use
`A:10:1:PEO-1.0`. Note that now the macro is called A. If we want instead to have a random
distribution of PS and PEO with equal probabilities, we would use `A:10:1:PEO-0.5,PS-0.5`. In
that case, PS and PEO are randomly distributed along the graph with equal probability. In
contrast, if we want to generate a polymer that is tree-like and consists of
poly(propylene imine) (PPI) residues, we would use `A:4:2:PPI-1.0`.

Once the macros are defined they can be combined to form a residue graph. To do so, first the
macros that are to be used have to be enumerated using the `-seq` flag. This specifies the
order of the macros in the final graph. Subsequently, the connections between the macros have
to be defined using the `-connects` flag and the following syntax:

```
<seq_position_macro_A>,<seq_position_macro_B>-<connect_residueA>,<connect_residueB>
```

In this syntax, `<seq_position>` is the position index of the macro provided to the `-seq`
flag, and `<connect_residue>` is the residue ID of the residue that forms the connection
between macros A and B, respectively. In addition to these basic functionalities, the program
also allows you to attach tags to each residue in the sequence with the `-label` option. In
this way, for example, nodes can be labeled with their chirality/tacticity. There is also the
possibility to modify end-groups. Both options are described in more detail in the help menu of
the program. As polyply parses networkx-style JSON files, users can write their own protocols
for making residue graphs or edit them manually as well.
