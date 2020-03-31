#!/bin/bash

input="sens0112.dat"
output="sens_numbered.dat"

# delete temporary files
test -e tmp_numbered_full && rm tmp_numbered_full

# number lines
#cat "${input}" | nl -b a  > tmp_numbered_full

# do not number lines
cat "${input}" > tmp_numbered_full

# assemble the sed expression to delete a range of lines
# write in the global string "$allstr"
function get_delete_string()
{
    lower_range="$1"
    upper_range="$2"
    echo $lower_range
    echo $upper_range

    dots=$lower_range","$((lower_range+2))
    lower_range=$((lower_range+3))
    echo $dots
    echo $lower_range
    echo $upper_range
    delstr="$dots"'s/^.*$/\t\t.../;'
    delstr=$delstr"$lower_range,"$upper_range"d;"
    allstr=$allstr$delstr
}

allstr=""
# prepare sed expression to delete the following lines
get_delete_string 5 1598

# execute the sed expression
sed $allstr tmp_numbered_full >> "${output}"

rm tmp_numbered_full
