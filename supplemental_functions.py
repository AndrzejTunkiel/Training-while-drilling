# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:33:25 2020

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no
"""


import numpy as np



def sampling_fix(df,name,start,stop,radius,medianFilter,plot):
    #Filter dataset based on depth range
    df = df[(df['Measured Depth m'] > start) & (df['Measured Depth m'] < stop) ]
    #remove NaNs from dataset
    df = df[np.isfinite(df[name])]
    X = df['Measured Depth m']
    
    #reshape the depth to matcch regressor requirements
    X = X.values.reshape(X.shape[0],1)
    from sklearn.neighbors import  RadiusNeighborsRegressor
    #define regressor with provided radius
    reg = RadiusNeighborsRegressor(radius=radius, weights='uniform')
    
    #apply median filter with back filling (to remove NaNs at the beginning of dataset)
    y = df[name].rolling(medianFilter).median().bfill()
    
    #fit regressor
    reg.fit(X, y)
    
    #check if plotting was required or should the model be returned
    if plot == 0:
        return reg
    else:
        import matplotlib.pyplot as plt
        #plot the chart. Original data is plotted as well as the regression data.
        plt.scatter(X,y, label=name)
        plt.plot(X, reg.predict(X),c='r',label="prediction")
        plt.legend()
        plt.show()



def prepareinput_nominal(data, memory_local):
    memory = memory_local
    stack = []
    for i in range(memory):
        stack.append(np.roll(data, -i))

    X_temp = np.hstack(stack)


    X_min = X_temp[:,0]
    X = X_temp-X_min[:,np.newaxis] #here the function will move the values to zero, used for inclination data processing
    return X


def prepareinput(data, memory_local):
    memory = memory_local
    stack = []
    for i in range(memory):
        stack.append(np.roll(data, -i))

    X_temp = np.hstack(stack)


    
    X = X_temp
    return X


def prepareinput_nozero(data, memory_local, predictions):
    memory = memory_local
    stack = []
    for i in range(memory+predictions):
        stack.append(np.roll(data, -i))

    X = np.hstack(stack) 
    return X


def prepareoutput(data, memory_local, predictions):
    memory = memory_local
    stack = []
    for i in range(memory, memory+predictions):
        stack.append(np.roll(data, -i))

    X = np.hstack(stack)
    return X

def expandaxis (var):
    var = np.expand_dims(var, axis=1)
    return var
    
def mymanyplots(epoch, data, model):
    
    import matplotlib.pyplot as plt
    [X1, X2, X3, X4, y, X1_train,X_train, X_test, X1_test, border1, border2, y_train, y_test, memory, y_temp, predictions] = data

    Xtrain = model.predict(X_train)
    Xtest = model.predict(X_test)
    
    

    for i in range(1):
        shape = (7,1)
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace = 1)
        ax1 = plt.subplot2grid(shape, (0,0), rowspan=4)    
        ax2 = plt.subplot2grid(shape, (4,0), sharex=ax1)
        ax3 = plt.subplot2grid(shape, (5,0), sharex=ax1)    
        ax4 = plt.subplot2grid(shape, (6,0), sharex=ax1)
        
        for ax in fig.axes:
            plt.setp(ax.get_xticklabels(), visible=False)


        known_attributes = ['Average Surface Torque kN.m', 'Average Rotary Speed rpm', 'Rate of Penetration m/h']

        i=0
        for axe in fig.axes:
            if i == 0:
                tr = np.random.randint(0, border1)
                axe.plot(np.arange(memory,memory+predictions,1),y_train[tr],linewidth=5,alpha=0.5,c='b', label='training input')
                axe.plot(np.arange(0,memory,1),X1_train[tr], linewidth=5,alpha=0.5, c='g' , label="training expected")
                axe.plot(np.arange(memory,memory+predictions,1),Xtrain[tr],c='r', label='training predicted')
                axe.set_title('Training results')
                axe.set_facecolor('xkcd:light blue')
                axe.legend()

            else:

                axe.plot(np.arange(0,memory+predictions,1),X_train[1][tr,:,i-1],label=known_attributes[i-1]) 
                axe.set_facecolor('xkcd:ivory')
                axe.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
            i = i + 1
        plt.show()

    for i in range(3):
        shape = (7,1)
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace = 1)
        ax1 = plt.subplot2grid(shape, (0,0), rowspan=4)    
        ax2 = plt.subplot2grid(shape, (4,0), sharex=ax1)
        ax3 = plt.subplot2grid(shape, (5,0), sharex=ax1)    
        ax4 = plt.subplot2grid(shape, (6,0), sharex=ax1)
        
        for ax in fig.axes:
            plt.setp(ax.get_xticklabels(), visible=False)

        known_attributes = ['Average Surface Torque kN.m', 'Average Rotary Speed rpm', 'Rate of Penetration m/h']

        i=0
        for axe in fig.axes:
            if i == 0:
                tr = np.random.randint(0, border2 - border1)
                axe.plot(np.arange(memory,memory+predictions,1),y_test[tr],linewidth=5,alpha=0.5,c='b', label='testing input')
                axe.plot(np.arange(0,memory,1),X1_test[tr], linewidth=5,alpha=0.5, c='g' , label="testing expected")
                axe.plot(np.arange(memory,memory+predictions,1),Xtest[tr],c='r', label='testing predicted')
                axe.set_title('Testing results')
                axe.set_facecolor('xkcd:light grey')
                axe.legend()

            else:

                axe.plot(np.arange(0,memory+predictions,1),X_test[1][tr,:,i-1],label=known_attributes[i-1]) 
                axe.set_facecolor('xkcd:ivory')
                axe.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
            i = i + 1
        plt.show()

   