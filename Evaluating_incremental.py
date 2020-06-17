# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:36:57 2020

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no
"""

import logging

from logging import debug
from logging import info
from logging import warning
from logging import error
from logging import critical

logging.basicConfig(format='%(asctime)s - %(levelname)s:%(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', 
                    #filename='app.log',
                    level=logging.WARNING)


#%%

import warnings
warnings.filterwarnings('ignore')

from numpy.random import seed
seed(0)
import tensorflow as tf

tf.random.set_seed(0)

info(tf.__version__)

import pandas as pd
import numpy as np


#%%

from training_delta_best import train_delta
from training_nominal import train_nominal
from training_nominal_middleVal import train_nominal_middleVal

from supplemental_functions import expandaxis

begin = 500
middle = 800
end = 843

model_array = []
val_loss_array = []
hypers_array = []


percent_drilled = np.arange(15,81,1)

#percent_drilled = [60]
matrix_size = len(percent_drilled)




newdim = np.full((matrix_size,),3)
start = np.full((matrix_size,), begin)
stop = np.full((matrix_size,), end)
inc_layer1 = np.full((matrix_size,), 371) #was256
inc_layer2 = np.full((matrix_size,), 2) #np.full((matrix_size,), 48) 
#gaussian noise now, divided by 1000.
data_layer1 = np.full((matrix_size,), 1) 
data_layer2 = np.full((matrix_size,), 1) #drop2
dense_layer = np.full((matrix_size,), 8) #was139 
                                        #np.arange(139-step, 139+step+1, step) 
range_max =np.full((matrix_size,), 1)  #DISABLED
memory =  np.full((matrix_size,), 100) #np.arange(70, 101, step) 
                                    # np.full((matrix_size,), 200) 
                                    #was86 #np.arange(86-step, 86+step+1, step) 
predictions = np.full((matrix_size,), 100)
drop1 = np.full((matrix_size,), 50)
drop2 = np.full((matrix_size,), 0) #np.random.randint(50,90,size=matrix_size)
lr = np.full((matrix_size,), 40) #was 16
bs = np.full((matrix_size,), 32)
ensemble_count = np.full((matrix_size,),1)

inputs = [newdim, percent_drilled, start, stop, inc_layer1,
                    inc_layer2,data_layer1,data_layer2,dense_layer,
                    range_max, memory, predictions, drop1, drop2, lr, bs, ensemble_count]

for loc, val in enumerate(inputs):
    inputs[loc] = expandaxis(val)
    

hypers = np.hstack(inputs)

ID = np.random.randint(0,999999)

histories = []
val_loss_array = []
aae_array = []
for i, val in enumerate(hypers):
    print (f'Evaluating {i+1}/{len(hypers)}')
   
    val_loss, test_loss, aae = train_delta(val)
    aae_array.append(aae)
    
# =============================================================================
#     val_loss_array.append(val_loss)
#     hypers_array.append(i)
#     print (i)
#     print (val_loss)
#     output = np.append(hypers_array, expandaxis(val_loss_array), axis=1)
#     output = pd.DataFrame(output,columns=["PCA dim count",
#                                           "percentage_drilled", "start",
#                                           "stop", "inc_layer1","inc_layer2",
#                                           "data_layer1","data_layer2",
#                                           "dense_layer"," range_max",
#                                           " memory"," predictions","drop1",
#                                           "drop2", "val loss"])
# =============================================================================
# =============================================================================
#     try:
#         #print (output)
#         output.to_csv("FastICA " + str(ID) +".csv")
#     except:
#         print("File opened?")
# 
# =============================================================================

#%%
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(aae_array)
np.save('delta ave no lottery.npy', aae_array)