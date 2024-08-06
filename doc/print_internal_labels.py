#!/usr/bin/env python
"""
These should be the labels that we can refer to with :ref:`[ENTRY]`
"""
import pickle
with open("./_build/doctrees/environment.pickle", "rb") as f:
    dat = pickle.load(f)

for x in dat.domaindata['std']['labels'].keys():
    print(x)
