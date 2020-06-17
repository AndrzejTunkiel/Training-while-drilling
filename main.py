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

from training_delta import train_delta
from training_nominal import train_nominal
from training_nominal_middleVal import train_nominal_middleVal

from supplemental_functions import expandaxis

begin = 500
middle = 800
end = 843

model_array = []
val_loss_array = []
hypers_array = []


#percent_drilled = np.arange(15,81,1)

percent_drilled = [50]
matrix_size = len(percent_drilled)




newdim = np.full((matrix_size,),3)
start = np.full((matrix_size,), begin)
stop = np.full((matrix_size,), end)
inc_layer1 = np.full((matrix_size,), 111) #was256
inc_layer2 = np.full((matrix_size,), 38) #np.full((matrix_size,), 48) 
#gaussian noise now, divided by 1000.
data_layer1 = np.full((matrix_size,), 37) 
data_layer2 = np.full((matrix_size,), 1) #drop2
dense_layer = np.full((matrix_size,), 50) #was139 
                                        #np.arange(139-step, 139+step+1, step) 
range_max =np.full((matrix_size,), 1)  #DISABLED
memory =  np.full((matrix_size,), 103) #np.arange(70, 101, step) 
                                    # np.full((matrix_size,), 200) 
                                    #was86 #np.arange(86-step, 86+step+1, step) 
predictions = np.full((matrix_size,), 100)
drop1 = np.full((matrix_size,), 1)
drop2 = np.full((matrix_size,), 16) #np.random.randint(50,90,size=matrix_size)
lr = np.full((matrix_size,), 16) #was 16
bs = np.full((matrix_size,), 128)

inputs = [newdim, percent_drilled, start, stop, inc_layer1,
                    inc_layer2,data_layer1,data_layer2,dense_layer,
                    range_max, memory, predictions, drop1, drop2, lr, bs]

for loc, val in enumerate(inputs):
    inputs[loc] = expandaxis(val)
    

hypers = np.hstack(inputs)

ID = np.random.randint(0,999999)

histories = []
for i in hypers:
    print("###")
    #val_loss = train_delta(i)
    
    #val_loss = train_nominal_middleVal(i)
    #print(val_loss)
    val_loss, history = train_nominal(i)
    histories.append(history)
    #model_array.append(model)
    
    print (val_loss)
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
kernels = ["zeros", "ones", "random_normal", "random_uniform",
               "glorot_normal", "glorot_uniform", "orthogonal", "identity"]

for i in range(len(kernels)):
    
    plt.plot(np.log(histories[i].history['val_loss']), label=kernels[i])
    plt.legend()
    plt.title("Val loss, kernels")

plt.show()

#%%
def minimize_me(data_layer1,dense_layer,drop1,drop2,inc_layer1,inc_layer2,lr,bs):
    newdim = 3
    start = begin
    stop = end
    range_max = 1
    percent_drilled = 40
    data_layer2 = 1
    predictions = 100
    memory = 100
    
    percent_drilled = 30
    params = [newdim, percent_drilled, start, stop, inc_layer1,
                    inc_layer2,data_layer1,data_layer2,dense_layer,
                    range_max, memory, predictions, drop1, drop2, lr, bs]
    params = np.array(params,dtype=int)
    result1, _ = train_nominal(params)
    
    percent_drilled = 55
    params = [newdim, percent_drilled, start, stop, inc_layer1,
                    inc_layer2,data_layer1,data_layer2,dense_layer,
                    range_max, memory, predictions, drop1, drop2, lr, bs]
    params = np.array(params,dtype=int)
    result2, _ = train_nominal(params)
    
    percent_drilled = 80
    params = [newdim, percent_drilled, start, stop, inc_layer1,
                    inc_layer2,data_layer1,data_layer2,dense_layer,
                    range_max, memory, predictions, drop1, drop2, lr, bs]
    params = np.array(params,dtype=int)
    result3, _ = train_nominal(params)
    
    result = (result1 + result2 + result3)/3
    return -result
    
from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'data_layer1' : (1,64),
    'dense_layer' : (8,512),
    'drop1' : (0,50),
    'drop2' : (0,50),
    'inc_layer1' : (8,512),
    'inc_layer2' : (2,256),
    'lr' : (1,40),
    'bs': (1, 128)}

optimizer = BayesianOptimization(
    f=minimize_me,
    pbounds=pbounds,
    random_state=1,
)

#%%

optimizer.probe(
    params={'data_layer1' : (20),
    'dense_layer' : (110),
    'drop1' : (25),
    'drop2' : (25),
    'inc_layer1' : (111),
    'inc_layer2' : (38),
    'lr' : (16),
    'bs': (32)},
    lazy=True,
)


#%%

optimizer.maximize(
    init_points=50,
    n_iter=20,
)
print(optimizer.max)

for i in range(50):
    optimizer.maximize(
        n_iter=25,
    )
    print(optimizer.max)
#%%
optimizer.maximize(
    n_iter=1000,
)
#%%
print(optimizer.max)