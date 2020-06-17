# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:41:07 2020

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('F9ADepth.csv')

import seaborn as sns
#%%

df = df[df['Measured Depth m']<848]
df = df[df['Measured Depth m']>500]


plt.figure(figsize=(5,4))
plt.scatter(x = df['Measured Depth m'],
            y = df['MWD Continuous Inclination dega'],
            c = "black",
            s=1)
plt.ylabel('MWD Continuous Inclination [deg]')
plt.xlabel('Measured Depth [m]')
plt.grid()
plt.savefig('wellprofile.pdf')