# Quick Start

This page gives an overview of the most important commands. For in-depth examples, see the
**Tutorials**.

## Polymer library

Some macromolecules are implemented in our library for a range of different force-fields.
To get a list of all libraries available, run:

```bash
polyply -list-lib
```

To get a detailed list of all the individual macromolecules:

```bash
polyply -list-blocks <Force Field Name>
```

!!! note
    While you can combine fragments from different libraries (e.g. the martini3 polymer
    library with martini3 proteins), we cannot guarantee that all the links are present to
    make a meaningful itp. All blocks within a library can safely be combined. For more
    information on how to implement links between different blocks, see
    [Writing .ff Input Files](reference/writing-ff-input-files.md).

See the [Polymer Library](reference/polymer-library.md) page for the full catalogue of
currently available parameters.

## itp file generation

To generate a linear polymer chain using parameters provided in the library, run:

```bash
polyply gen_params -lib <library_name> -name <name> -seq <monomer:#number> -o <name_outfile.itp>
```

For more information on how to generate itp files for more complex polymers, or how to
combine them with existing itp files, see the **Tutorials** and the
[Syntax Reference](reference/build-file-syntax.md).

## Initial structure generation

To generate an initial structure, run:

```bash
polyply gen_coords -p <top> -o <name_outfile.gro> -name <name of molecule> -dens <density>
```

or:

```bash
polyply gen_coords -p <top> -o <name_outfile.gro> -name <name of molecule> -box <x> <y> <z>
```

In order to append coordinates to an already existing coordinate file, run:

```bash
polyply gen_coords -p <top> -o <name_outfile.gro> -name <name of molecule> -c <init_coords.gro> -box <x> <y> <z>
```

or:

```bash
polyply gen_coords -p <top> -o <name_outfile.gro> -name <name of molecule> -c <init_coords.gro> -dens <density>
```

!!! note
    At the moment, polyply can only generate disordered structures of polymers. All
    molecules that have secondary structure (e.g. DNA, proteins) cannot be generated
    accurately. Chirality is also not taken into account by default; all polymers are
    atactic unless a dihedral specifies the chirality.
