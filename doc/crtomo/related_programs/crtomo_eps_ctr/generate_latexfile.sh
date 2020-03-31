#!/bin/bash

rm eps_tmp
rm eps.ctr_numbered

echo "{\tiny
\begin{verbatim}
" > eps.ctr_numbered

# number lines
cat eps.ctr | nl -b a  >> eps_tmp

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
    delstr="$dots"'s/^.*$/\t\t.../;'
    delstr=$delstr"$lower_range,"$upper_range"d;"
    allstr=$allstr$delstr
}

allstr=""
get_delete_string 3 526
get_delete_string 532 1055
get_delete_string 1061 1584
get_delete_string 1590 2113
get_delete_string 2119 2642
get_delete_string 2648 3171
get_delete_string 3177 3700

echo $allstr
sed $allstr eps_tmp >> eps.ctr_numbered

#sed '7s/^.*$/.../;8,522d;536s/^.*/.../;537,1051d' eps_tmp >> eps.ctr_numbered

echo "\end{verbatim}
}
" >> eps.ctr_numbered

