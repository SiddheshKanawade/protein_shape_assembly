mol delete all
set nameid "521p"
mol new ${nameid}.pdb
set sel [atomselect top "protein and name CA"]
set structure [$sel get structure]
puts $structure
set resid [$sel get resid]
set max [tcl::mathfunc::max {*}$resid]

set startindex {}
set endindex {}
set end 0
for {set i 0} {$i < $max} {incr i} {
      #for the secondary structure if i and i+1
        set id1 $i
        set id2 [expr $i+1]
	set one [lindex $structure $id1]
        set two [lindex $structure $id2]
     #End

if { $one == $two } {
    set end [expr $i+1]
} else {
 lappend endindex [expr $id2 - 1]

}

}

#calculate the total numbers of secondary structures
set lengthsecst [llength $endindex]
lappend startindex 0
for {set i 0} {$i < $lengthsecst-1} {incr i} {
        set j [lindex $endindex $i]
	lappend startindex [expr $j+1]
}

puts $startindex
puts $endindex
axes location Off
##Creating images
for {set i 0} {$i < $lengthsecst-1} {incr i} {
	set startpos [lindex $startindex $i]
        set endpos [lindex $endindex $i]
		set protein [atomselect top "residue $startpos to $endpos"]
		$protein writepdb ${nameid}_${startpos}_${endpos}.pdb
        #mol modselect 0 0 residue $startpos to $endpos
	#mol modstyle 0 0 NewCartoon 0.300000 10.000000 4.100000 0
        #render snapshot ${nameid}_${i}.png explorer %s
        #render Tachyon ${nameid}_$i "/usr/local/lib/vmd/tachyon_LINUXAMD64" -aasamples 12 %s -format BMP -res 1200 1200  -o %s.bmp
        #mv pattern1.bmp ${nameid}_$i.bmp
}

#mol modselect 0 0 residue 0 to $max
#mol modstyle 0 0 NewCartoon 0.300000 10.000000 4.100000 0
#render snapshot ${nameid}.png explorer %s
#render Tachyon ${nameid} "/usr/local/lib/vmd/tachyon_LINUXAMD64" -aasamples 12 %s -format BMP -res 1200 1200  -o %s.bmp
#mv pattern1.bmp ${nameid}.bmp

puts "***************DONE*********************"

exit
