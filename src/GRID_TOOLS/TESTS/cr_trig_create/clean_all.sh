#!/bin/bash

find . -name "test.sh" -execdir bash -c "test -d grid && rm -r grid" \;
