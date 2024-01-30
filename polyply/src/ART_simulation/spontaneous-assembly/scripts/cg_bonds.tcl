
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #                                                                           #
  #                                                                           #
  #   --- DISCLAIMER (by Clement Arnarez, C.Arnarez@rug.nl):                  #
  #                                                                           #
  #   This script is largely inspired by the one written by Nicolas Sapay     #
  #   (hum... less and less true, after the recent complete refont of the     #
  #   code), available on the GROMACS website.                                #
  #                                                                           #
  #   Initially written to read Martini and ElNeDyn topologies, it seems to   #
  #   work for any couple of conformation/topology file (even for all atom    #
  #   systems apparently) generated with the GROMACS package.                 #
  #                                                                           #
  #   As always, you can modify, redistribute and make everything you want    #
  #   with these few lines of code; if you write major improvement, please    #
  #   let me know/test it!                                                    #
  #                                                                           #
  #                                                                           #
  #   --- ORIGINAL DISCLAIMER (by Nicolas Sapay):                             #
  #                                                                           #
  #   Somewhere in there...:                                                  #
  #   http://lists.gromacs.org/pipermail/gmx-users/2009-October/045935.html   #
  #   ... and there:                                                          #
  #   http://www.gromacs.org/Developer_Zone/Programming_Guide/VMD             #
  #                                                                           #
  #   TCL Script to visualize CG structures in VMD                  # # # # # #
  #   Version 3.0                                                   #       #
  #   Adapted from vmds                                             #     #
  #                                                                 #   #
  #                                                                 # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #





### ------------------------------------------------------------------------------------------------ USAGE

# shows usage
proc cg_bus { } { cg_bonds_usage }
proc cg_bonds_usage { } {
  puts ""
  puts " USAGE"
  puts "-------"
  puts ""
  puts "These few lines are given by the \"cg_bonds_usage\" command."
  puts ""
  puts "Draw bonds for specific molecules."
  puts "   cg_bonds \[OPTIONS\]"
  puts ""
  puts "If you ask to this script to draw Martini/elastic network from a .tpr file, it will use the \"gmxdump\" executable compiled with gromacs. By default, it looks for it in the \"/usr/bin\" directory; you can precise another path with specific option (see below). If you ask to read a .top file, it will read the paths to the .itp files from the \"#include\" statements; make sure those paths are valid..."
  puts ""
  puts "Options and default values:"
  puts "   -molid     \"top\"                VMD-defined ID of the molecule to process"
  puts "   -gmx       /usr/bin/gmxdump     path to the \"gmxdump\" executable"
  puts "   -tpr       topol.tpr            path to the simulation file (.tpr)"
  puts "   -top       topol.top            path to the system topology files (.top linking to .itp)"
  puts "   -topoltype \"martini\"            type of topology(martini|elastic|elnedyn)"
  puts "   -net       \"martini\"            network to draw (martini|elastic|both)"
  puts "   -bndrcnstr \"both\"               draw bonds AND/OR constraints (bonds|constraints|both)"
  puts "   -cutoff    7.0                  cutoff for bonds/constraints \[angstroms\]"
  puts "   -color     \"red\"                color (color name or VMD-defined ID) of elastic bonds"
  puts "   -mat       \"Opaque\"             material for elastic bonds"
  puts "   -rad       0.2                  radius of elastic bonds"
  puts "   -res       6                    resolution of elastic bonds"
  puts ""
  puts "In most of the cases, if a \"classical\" cutoff is used for the elastic network (0.9nm), more than 12 elastic \"bonds\" per bead are defined and VMD refuses to draw them. BUT this script replaces bonds by cylinders, and is able to draw the entire elastic network. Note that you will have to modify the default cutoff value to see all the \"bonds\" defined by the elastic network. These drawings are not dynamic, you will need to re-draw the entire network each step of a trajectory."
  puts ""
  puts "Delete the Martini network:"
  puts "   cg_delete_martini_bonds \[OPTION\]"
  puts ""
  puts "Delete the elastic network:"
  puts "   cg_delete_elastic_bonds \[OPTION\]"
  puts ""
  puts "Shortcut (delete everything, including bonds/constraints lists):"
  puts "   cg_delete_all \[OPTION\]"
  puts ""
  puts "The only option for the previous three commands is the VMD-defined ID of the molecule to process (via -molid, default is \"top\")."
  puts ""
  puts "Someone (...) said the names of the functions were too long; you can find some shortcuts here:"
  puts "   cg_bonds_usage            cg_bus"
  puts "   cg_delete_martini_bonds   cg_dmb"
  puts "   cg_delete_elastic_bonds   cg_deb"
  puts "   cg_delete_all             cg_dab"
  puts ""
  puts " EXAMPLES"
  puts "----------"
  puts ""
  puts "   cg_bonds -tpr mini.tpr"
  puts "   cg_bonds -gmx /usr/local/gromacs-4.0.2/bin/gmxdump -topoltype \"martini\" -top system.top -cutoff 6.2"
  puts "   cg_bonds -topoltype \"elastic\" -res 10 -cutoff 12.0 -mat Transparent -color 12"
  puts "   cg_dmb -molid 1"
  puts ""
}

# first load
cg_bonds_usage

###





### ------------------------------------------------------------------------------------------------ UTILS

# check if file exists
proc file_exists { file } { if { [file exists $file] } { return -code 0 } else { return -code 1 "\nError: file $file does not exist.\n" } }

# add bond/constraint to the list table
proc add_link { molecule_id first_bead second_bead LNK } {
  upvar $LNK links
  if { [info exists links($molecule_id,$first_bead)] } { lappend links($molecule_id,$first_bead) $second_bead } else { set links($molecule_id,$first_bead) $second_bead }
  if { [info exists links($molecule_id,$second_bead)] } { lappend links($molecule_id,$second_bead) $first_bead } else { set links($molecule_id,$second_bead) $first_bead }
}

# add list of beads linked to the current one
proc link_bead { total_occurences bead_numbers LNK cutoff molecule_id start_at network } {
  upvar $LNK links
  for { set occurence 0 } { $occurence < $total_occurences } { incr occurence } {
    set bead_number [dict get $bead_numbers $molecule_id]
    for { set bead 0 } { $bead < $bead_number } { incr bead } {
      if { [info exists links($molecule_id,$bead)] } {
        set bead_index [expr $bead+$start_at+$occurence*$bead_number]
        set linked_beads {}
        foreach linked_bead $links($molecule_id,$bead) {
          set linked_bead_index [expr $linked_bead+$start_at+$occurence*$bead_number]
          if { [lsearch $linked_beads $linked_bead_index] == -1 && [measure bond "$bead_index $linked_bead_index"] < $cutoff } { lappend linked_beads $linked_bead_index }
        }
        lappend network $linked_beads
      } else {
        lappend network {}
      }
    }
  }
  return $network
}

###





### ------------------------------------------------------------------------------------------------ PARSING FILES
#
# Martini:
#   1) backbone-backbone bonds
#   2) backbone-sidechain bonds
#   3) sidechain-sidechain bonds
#   4) short elastic bonds
#   5) long elastic bonds
#   6) sidechain-sidechain constraints 
#   7) backbone-sidechain constraints
#
# Martini + elastic network:
#   1) backbone-backbone bonds
#   2) elastic network bonds           (.itp file)
#   3) backbone-sidechain bonds
#   4) sidechain-sidechain bonds
#   5) short elastic bonds
#   6) long elastic bonds
#   7) sidechain-sidechain constraints 
#   8) backbone-sidechain constraints
#   9) elastic network bonds           (.tpr file)
#
# Martini + ElNeDyn:
#   1) backbone-backbone bonds
#   2) elnedyn bonds                   (.itp/.tpr file)
#   3) backbone-sidechain bonds
#   4) backbone-backbone constraints
#   5) backbone-sidechain and sidechain-sidechain constraints

# (gmx)dumps a coarse-grained .tpr file and extracts bonds and constraints
proc parse_gmxdump_output { molid gmxdump tpr topology_type BD ELAST CNSTR } {

  # system topology
  set molecules {}
  set occurences {}
  set bead_numbers [dict create]
  # total number of beads in the system
  set total_bead_number 0
  # molecules
  set molecule_id 0
  set occurence 0
  set bead_number 0
  # current bond/constraint
  set first_bead 0
  set second_bead 0
  set previous_first_bead 0
  set previous_second_bead 0
  # type of bonds read
  set which_type_of_link 0

  # bonds/constraints
  upvar $BD bonds
  upvar $ELAST elastic
  upvar $CNSTR constraints

  # opens the file (dumps the .tpr) and read it
  set tpr [open "| $gmxdump -s $tpr 2> /dev/null | grep -F -e \#atoms -e \#beads -e moltype -e \#molecules -e \(BONDS\) -e \(CONSTR\) -e \(HARMONIC\)" "r"]
  while { [gets $tpr line] > 0 } {

    # total number of beads
    regexp {^\s+\#atoms+\s+=+\s+(\d+)} $line 0 total_bead_number
    if { $total_bead_number != 0 && $total_bead_number != [molinfo $molid get numatoms] } { return -code 1 "Error: the TPR file and VMD doesn\"t define the same number of beads! Please provide the good TPR file." }

    # molecule id/name, occurence, number of beads
    if { [regexp {^\s+moltype\s+=\s+(\d+)} $line 0 molecule_id] } { lappend molecules $molecule_id }
    if { [regexp {^\s+\#molecules\s+=\s+(\d+)} $line 0 occurence] } { lappend occurences $occurence }
    if { [regexp {^\s+\#atoms_mol\s+=\s+(\d+)} $line 0 bead_number] } { dict set bead_numbers $molecule_id $bead_number }

    # extracts bonds and constraints for the current molecule
    if { [regexp {^\s+moltype\s+\((\d+)\):} $line 0 molecule_id] } {
      # current bond/constraint
      set first_bead 0
      set second_bead 0
      # type of bonds read
      set which_type_of_link 0
    }
    # bonds
    if { [regexp {\(BONDS\)\s+(\d+)\s+(\d+)} $line 0 first_bead second_bead] } {
      # boolean to know which bonds are read
      if { $first_bead < $previous_first_bead && $second_bead < $previous_second_bead } { incr which_type_of_link }
      # actual bond
      if { $topology_type == "martini" || $topology_type == "elastic" } { add_link $molecule_id $first_bead $second_bead bonds } elseif { $topology_type == "elnedyn" } { if { $which_type_of_link == 1 } { add_link $molecule_id $first_bead $second_bead elastic } else { add_link $molecule_id $first_bead $second_bead bonds } }
    }
    # constraints
    if { [regexp {\(CONSTR\)\s+(\d+)\s+(\d+)} $line 0 first_bead second_bead] } {
      # actual constraint
      add_link $molecule_id $first_bead $second_bead constraints
    }
    # harmonic elastic bonds
    if { [regexp {\(HARMONIC\)\s+(\d+)\s+(\d+)} $line 0 first_bead second_bead] } {
      # actual harmonic elastic bond
      add_link $molecule_id $first_bead $second_bead elastic
    }

    # sets previous atom indexes (for later comparison)
    set previous_molecule_id $molecule_id
    set previous_first_bead $first_bead
    set previous_second_bead $second_bead

  }

  # closes file
  close $tpr


  # return results
  return [list $molecules $occurences $bead_numbers]

}



# reads .top-related .itp files and extract bonds and constraints
proc parse_itp { itp topology_type bead_numbers BD ELAST CNSTR } {

  # bonds/constraints
  upvar $BD bonds
  upvar $ELAST elastic
  upvar $CNSTR constraints
  # boolean
  set read_moleculetype "False"
  set read_atoms "False"
  set read_bonds "False"
  set read_constraints "False"
  # molecules
  set molecule_id ""
  set previous_molecule_id ""
  set bead_number 0
  # current bond/constraint
  set first_bead 0
  set second_bead 0
  set previous_first_bead 0
  set previous_second_bead 0
  # type of bonds read
  set which_type_of_link 0

  # opens the .itp file
  set itp [open $itp "r"]
  while 1 {
    gets $itp line
    set line [string trim $line]
    if [eof $itp] break
    if { [string bytelength $line] > 0 && [string first ";" [string trim $line] 0] != 0 && [string first "#" [string trim $line] 0] != 0 } {

      # read nothing
      if { [string first "\[" $line 0] == 0 && [string first "angles" $line 0] > -1 } {
        set read_moleculetype "False"
        set read_atoms "False"
        set read_bonds "False"
        set read_constraints "False"
      }

      # reads constraints
      if { $read_constraints == "True" } {
        # bead indexes
        regexp {(\d+)\s+(\d+)} $line 0 first_bead second_bead
        set first_bead [expr $first_bead-1]
        set second_bead [expr $second_bead-1]
        # actual constraints
        add_link $molecule_id $first_bead $second_bead constraints
      }
      if { [string first "\[" $line 0] == 0 && [string first "constraints" $line 0] > -1 } {
        set read_moleculetype "False"
        set read_atoms "False"
        set read_bonds "False"
        set read_constraints "True"
      }

      # reads bonds
      if { $read_bonds == "True" } {
        # bead indexes
        regexp {(\d+)\s+(\d+)} $line 0 first_bead second_bead
        set first_bead [expr $first_bead-1]
        set second_bead [expr $second_bead-1]
        # boolean to know which bonds are read
        if { $first_bead < $previous_first_bead && $second_bead < $previous_second_bead } { incr which_type_of_link }
        # actual bond
        if { $topology_type == "martini" } { add_link $molecule_id $first_bead $second_bead bonds } elseif { $topology_type == "elastic" || $topology_type == "elnedyn" } { if { $which_type_of_link == 1 } { add_link $molecule_id $first_bead $second_bead elastic } else { add_link $molecule_id $first_bead $second_bead bonds } }
      }
      if { [string first "\[" $line 0] == 0 && [string first "bonds" $line 0] > -1 } {
        set read_moleculetype "False"
        set read_atoms "False"
        set read_bonds "True"
        set read_constraints "False"
      }

      # reads atom number
      if { $read_atoms == "True" } { if { [regexp {(\d+)\s+(.*)\s+(\d+)} $line 0] } { incr bead_number } }
      if { [string first "\[" $line 0] == 0 && [string first "atoms" $line 0] > -1 } {
        set read_moleculetype "False"
        set read_atoms "True"
        set read_bonds "False"
        set read_constraints "False"
      }

      # reads molecule name
      if { $read_moleculetype == "True" } {
        regexp {(.*)\s+(\d+)} $line 0 molecule_id whatever
        set molecule_id [string tolower [string trim $molecule_id]]
      }
      if { [string first "\[" $line 0] == 0 && [string first "moleculetype" $line 0] > -1 } {
        # booleans
        set read_moleculetype "True"
        set read_atoms "False"
        set read_bonds "False"
        set read_constraints "False"
        # current bond/constraint
        set first_bead 0
        set second_bead 0
        # type of bonds read
        set which_type_of_link 0
        # bead number
        if { $bead_number > 0 } { dict set bead_numbers $molecule_id $bead_number }
        set bead_number 0
      }

      # sets previous atom indexes (for later comparison)
      set previous_molecule_id $molecule_id
      set previous_first_bead $first_bead
      set previous_second_bead $second_bead

    }
  }

  # last molecule read
  dict set bead_numbers $molecule_id $bead_number

  # closes file
  close $itp

  # return results
  return $bead_numbers

}



# reads .top file
proc parse_top { molid top topology_type BD ELAST CNSTR } {

  # system topology
  set molecules {}
  set occurences {}
  set bead_numbers [dict create]
  # boolean (occurences)
  set read_molecules "False"

  # bonds/constraints
  upvar $BD bonds
  upvar $ELAST elastic
  upvar $CNSTR constraints

  # path to the .top file
  set path [lreplace [split $top "/"] end end]
  # opens the .top file and read it
  set top [open $top "r"]
  while 1 {
    gets $top line
    set line [string trim $line]
    if [eof $top] break
     if { [string bytelength $line] > 0 && [string first ";" [string trim $line] 0] != 0 } {

      # reads include files
      if { [string first "#include" $line 0] > -1 } {
        set itp [string trim [lindex [split $line] 1] "\""]
        if { [string first "/" $line 0] > -1 } { file_exists "[join $path "/"]/$itp" } else { file_exists $itp }
        set bead_numbers [parse_itp $itp $topology_type $bead_numbers bonds elastic constraints]
      }
 
      # reads system topology (occurences)
      if { $read_molecules == "True" } {
        regexp {(.*)\s+(\d+)} $line 0 molecule_id occurence
        lappend molecules [string tolower [string trim $molecule_id]]
        lappend occurences $occurence
      }
      if { [string first "\[" $line 0] == 0 && [string first "molecules" $line 0] > -1 } { set read_molecules "True" }
 
   }
  }

  # closes file
  close $top

  # return results
  return [list $molecules $occurences $bead_numbers]

}

###





### ------------------------------------------------------------------------------------------------ GENERATES AND DRAWS NETWORKS

# generates bond lists
proc generate_bond_lists { molecules occurences bead_numbers BD ELAST CNSTR cutoff bonds_and_or_constraints } {

  # real indexes of molecules
  set start_at 0
  # martini/elastic networks
  set martini_network {}
  set elastic_network {}

  # bonds/constraints
  upvar $BD bonds
  upvar $ELAST elastic
  upvar $CNSTR constraints
  array unset bonds_constraints

  # joins bonds/constraints tables, when needed
  for { set molecule 0 } { $molecule < [llength $molecules] } { incr molecule } {
    set molecule_id [lindex $molecules $molecule]
    for { set bead 0 } { $bead < [dict get $bead_numbers $molecule_id] } { incr bead } {
      if { [info exists bonds($molecule_id,$bead)] && ($bonds_and_or_constraints == "bonds" || $bonds_and_or_constraints == "both") } { set bonds_constraints($molecule_id,$bead) $bonds($molecule_id,$bead) }
      if { [info exists constraints($molecule_id,$bead)] && ($bonds_and_or_constraints == "constraints" || $bonds_and_or_constraints == "both") } {
        if { [info exists bonds_constraints($molecule_id,$bead)] } { set bonds_constraints($molecule_id,$bead) [concat $bonds_constraints($molecule_id,$bead) $constraints($molecule_id,$bead)] } else { set bonds_constraints($molecule_id,$bead) $constraints($molecule_id,$bead) }
      }
    }
  }

  # generates list of vmd bonds
  for { set molecule 0 } { $molecule < [llength $molecules] } { incr molecule } {
    set molecule_id [lindex $molecules $molecule]
    set occurence [lindex $occurences $molecule]
    set martini_network [link_bead $occurence $bead_numbers bonds_constraints $cutoff $molecule_id $start_at $martini_network]
    set elastic_network [link_bead $occurence $bead_numbers elastic $cutoff $molecule_id $start_at $elastic_network]
    set start_at [expr $start_at+($occurence*[dict get $bead_numbers $molecule_id])]
  }

  # return results
  return [list $martini_network $elastic_network]

}



# generates and draws network
proc cg_bonds { args } {
  set args [join $args]
  set args [split $args]

  # default values
  set molid "top"
  set gmxdump "/usr/bin/gmxdump"
  set use_tpr "False"
  set tpr "topol.tpr"
  set use_top "False"
  set top "topol.top"
  set topology_type "martini"
  set network "martini"
  set bonds_and_or_constraints "both"
  set cutoff 7.0
  set color 3
  set material "Opaque"
  set radius 0.2
  set resolution 10
  # parses arguments
  foreach { n m } $args {
    if { $n == "-molid" } { set molid $m }
    if { $n == "-gmx" } { set gmxdump $m }
    if { $n == "-tpr" } {
      file_exists $m
      set use_tpr "True"
      set tpr $m
      set use_top "False"
      set top ""
    }
    if { $n == "-top" } {
      file_exists $m
      set use_tpr "False"
      set tpr ""
      set use_top "True"
      set top $m
    }
    if { $n == "-topoltype" } { set topology_type $m }
    if { $n == "-net" } { set network $m }
    if { $n == "-bndrcnstr" } { set bonds_and_or_constraints $m }
    if { $n == "-cutoff" } { set cutoff $m }
    if { $n == "-color" } { set color $m }
    if { $n == "-mat" } { set material $m }
    if { $n == "-rad" } { set radius $m }
    if { $n == "-res" } { set resolution $m }
  }

  # bonds/constraints 
  array unset bonds
  array unset elastic
  array unset constraints

  # parses chosen files, return topologies
  if { $use_tpr == "True" } { lassign [parse_gmxdump_output $molid $gmxdump $tpr $topology_type bonds elastic constraints] molecules occurences bead_numbers } elseif { $use_top == "True" } { lassign [parse_top $molid $top $topology_type bonds elastic constraints] molecules occurences bead_numbers }
  # generates bond lists for given molecule and cutoff
  lassign [generate_bond_lists $molecules $occurences $bead_numbers bonds elastic constraints $cutoff $bonds_and_or_constraints] martini_network elastic_network

  # draws martini network...
  if { $network == "martini" || $network == "both" } { [atomselect $molid "all" frame 0] setbonds $martini_network }
  # ... and/or elastic network (takes much more time than drawing martini network!)
  if { $network == "elastic" || $network == "both" } {
    for { set n 0 } { $n < [llength $elastic_network] } { incr n } {
      set indexes [lindex $elastic_network $n]
      for { set m 0 } { $m < [llength $indexes] } { incr m } {
        set start [lindex [[atomselect $molid "index $n"] get { x y z }] 0]
        set end [lindex [[atomselect $molid "index [lindex $indexes $m]"] get { x y z }] 0]
        set middle [vecadd $start [vecscale 0.5 [vecsub $end $start]]]
        graphics $molid color $color
        graphics $molid material $material
        graphics $molid cylinder $start $end radius $radius resolution $resolution
        graphics $molid sphere $start radius $radius resolution $resolution
      }
    }
  }

}

###





### ------------------------------------------------------------------------------------------------ DELETES MARTINI/ELNEDYN NETWORKS

# deletes all martini bonds
proc cg_dmb { args } { cg_delete_martini_bonds $args }
proc cg_delete_martini_bonds { args } {
  set args [join $args]
  set args [split $args]

  # parses argument
  set molid "top"
  foreach { n m } $args { if { $n == "-molid" } { set molid $m } }

  # total number of beads
  set bead_number [molinfo $molid get numatoms]

  # creates the bond list
  set bond_list {}
  for { set index 0 } { $index < $bead_number } { incr index } { lappend bond_list {} }

  # draws an empty list of bonds
  set all [atomselect $molid all frame 0]
  $all setbonds $bond_list

}



# deletes all elastic bonds (cylinders)
proc cg_deb { args } { cg_delete_elastic_bonds $args }
proc cg_delete_elastic_bonds { args } {
  set args [join $args]
  set args [split $args]

  # parses argument
  set molid "top"
  foreach { n m } $args { if { $n == "-molid" } { set molid $m } }

  # deletes cylinders
  graphics $molid delete all

}



# deletes all generated stuffs
proc cg_dab { args } { cg_delete_all $args }
proc cg_delete_all { args } {
  set args [join $args]
  set args [split $args]

  # parses argument
  set molid "top"
  foreach { n m } $args { if { $n == "-molid" } { set molid $m } }

  # deletes martini network
  cg_delete_martini_bonds $molid

  # deletes elastic network (cylinders)
  cg_delete_elastic_bonds $molid

}

###
