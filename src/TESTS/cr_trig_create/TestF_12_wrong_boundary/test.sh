#!/bin/bash

outdir="grid"
test -d "${outdir}" && rm -r "${outdir}"
cr_trig_create.py "${outdir}"

