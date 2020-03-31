#!/bin/bash

output="elem.dat_numbered"
tmp="tmp"

rm $output
rm $tmp

echo "{\tiny
\begin{verbatim}
" > $output

# number lines
cat elem.dat | nl -b a  >> $tmp

# delete lines:
# 7 - 522
# 536 - 1051
# 1065 - 1580
# 1590 - 2113
# 2648 - 3171
# 3177 - 3700

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
    delstr="$dots"'s/^.*$/         .../;'

    delstr=$delstr"$lower_range,"$upper_range"d;"
    allstr=$allstr$delstr
}

allstr=""
get_delete_string 6 864
get_delete_string  867 1664
get_delete_string 1667 1704
get_delete_string 1707 1784
get_delete_string 1787 1904


echo $allstr
sed "$allstr" $tmp >> $output

echo "\end{verbatim}
}
" >> $output

