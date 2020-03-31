#!/usr/bin/python
# -*- coding: utf-8 -*-
from crlab_py.mpl import *
import crlab_py.elem as elem
import numpy as np

x = np.arange(-10,10,0.05)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, np.arcsinh(x))

ax.axvline(1, color='r')
ax.axvline(-1, color='r')
ax.axvspan(-1,1, color='r', alpha=0.5)
fig.savefig('asinh.png', dpi=150)
