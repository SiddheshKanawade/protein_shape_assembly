# Load the PDB file
mol new 1a6f.pdb

# Create an atom selection for CA atoms
set sel [atomselect top "protein and name CA"]

# Get the residue information (resid) and secondary structure (structure)
set residues [$sel get {resid structure}]

# Close the atom selection after retrieving information
$sel delete

# Initialize variables for fragment creation
set current_fragment {}
set current_structure ""
# Initialize a counter for fragment numbering
set fragment_counter 1

# Procedure to save the current fragment as a PDB file
proc saveFragment {} {
    global current_fragment
    set output_file "fragment_[incr ::fragment_counter].pdb"
    writepdb $output_file $current_fragment
    puts "Fragment saved as $output_file"
    set current_fragment {}
}

# Loop through the residues and create fragments
foreach {resid structure} $residues {
    if {$current_structure eq ""} {
        set current_structure $structure
    }
    
    if {$structure eq $current_structure} {
        lappend current_fragment [atomselect top "resid $resid"]
    } else {
        # Save the current fragment and start a new one
        saveFragment
        set current_structure $structure
        lappend current_fragment [atomselect top "resid $resid"]
    }
}

# Save the last fragment
saveFragment

# Delete the loaded structure when done (optional)
mol delete top
