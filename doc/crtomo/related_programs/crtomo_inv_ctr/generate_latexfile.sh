#!/bin/bash

echo "{\tiny
\begin{verbatim}
" > inv.ctr_numbered

cat inv.ctr | nl -b a  >> inv.ctr_numbered

echo "\end{verbatim}
}
" >> inv.ctr_numbered

