# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:34:19 2020

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no
"""

from logging import info

def train_delta(matrix):
    
    import numpy as np
    import tensorflow as tf
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import (Dense, Dropout, GRU,
                                         Flatten, GaussianNoise, concatenate)
    
    from tensorflow.keras.models import load_model
    
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    
    import supplemental_functions
    
    from supplemental_functions import (sampling_fix, prepareinput,
     prepareinput_nozero, prepareoutput)
        

    tf.keras.backend.clear_session()
    [newdim,percent_drilled, start, stop, inc_layer1,inc_layer2,data_layer1,
     data_layer2,dense_layer,range_max,memory, predictions, drop1,
     drop2, lr, bs, ensemble_count] = matrix
    drop1 = drop1/100
    drop2 = drop2/100
    inc_layer2 = inc_layer2/1000
    lr= lr/10000
    percent_drilled = percent_drilled/100
    df = pd.read_csv('F9ADepth.csv')
    
    df_target = df.copy()
    
    droplist = ['nameWellbore','name', 'Pass Name unitless',
                'MWD Continuous Inclination dega', 'Measured Depth m',
                'MWD Continuous Azimuth dega',"Unnamed: 0","Unnamed: 0.1"  ]
    for i in droplist:
        df = df.drop(i,1)

    
    for i in list(df):
        if df[i].count() < 1000:
            del df[i]
            info(f'dropped {i}')
      
    start = start
    stop = stop
    step = 0.230876

    X = np.arange(start,stop,step)
    X = X.reshape(X.shape[0],1)


    X = np.arange(start,stop,step)
    X = X.reshape(X.shape[0],1)

    my_data1 = sampling_fix(df_target, 'MWD Continuous Inclination dega',start,stop,1.7,1,0).predict(X)
    
    
    data_array = []
    
    for i in list(df):
        sampled = sampling_fix(df_target, i,start,stop,1.7,3,0).predict(X)
        if np.isnan(np.sum(sampled)) == False:
            data_array.append(sampled)
            info(f'Using {i}')


    data_array = np.asarray(data_array)
    dftemp = pd.DataFrame()
    dftemp['dinc'] = my_data1
    dftemp['dinc'] = dftemp['dinc'].diff(1).rolling(3,center=True).mean()

    my_data1 = dftemp['dinc'].ffill().bfill()
    

    data_array = data_array.T
    
    pre_PCA_scaler = MinMaxScaler()
    data_array = pre_PCA_scaler.fit_transform(data_array)
    
    
    from sklearn.decomposition import PCA

# =============================================================================
#     pca = PCA().fit(data_array)
#     plt.plot(np.cumsum(pca.explained_variance_ratio_))
#     plt.xlabel('number of components')
#     plt.ylabel('cumulative explained variance');
#     
#     plt.show()
# =============================================================================
    
    sampcount = int(len(data_array)*percent_drilled)
    
    pca = PCA(n_components=newdim).fit(data_array[:sampcount])
    projected = pca.transform(data_array)
    
    
    my_data = []
    
    for i in range(newdim):
        my_data.append(projected[:,i])

    

    my_data1 = my_data1[:,np.newaxis]
    
    
    my_data_newaxis = []
    for i in my_data:
         my_data_newaxis.append(i[:,np.newaxis])


    temp_data1 = pd.DataFrame(my_data1.flatten())
    temp_data1 = pd.DataFrame(my_data1)


    range1 = temp_data1[0].diff(memory+predictions)
    
    range2 = np.amax(range1)

    
    
    RNN_scaler = MinMaxScaler()

    my_data1 = RNN_scaler.fit_transform(my_data1)
    
    my_data_scaled = []
    for i in my_data_newaxis:
        my_data_scaled.append(MinMaxScaler().fit_transform(i))



    X1 = prepareinput(my_data1, memory)

    Xdata = []
    
    for i in my_data_scaled:
        Xn = prepareinput_nozero(i,memory, predictions)
        Xdata.append(Xn)

    
    y_temp = prepareoutput(my_data1, memory, predictions)

    stack = []
    for i in range(memory):
        stack.append(np.roll(my_data1, -i))

    X_temp = np.hstack(stack)


    y = y_temp 

    data_length = len(my_data1)-memory-predictions

    testing_cutoff = 0.80

    border1 = int((data_length)*(percent_drilled*0.8))
    border2 = int((data_length)*(percent_drilled))
    border3 = int((data_length)*(percent_drilled+0.2))

    X1_train =  X1[:border1]
    X1_test = X1[border1:border2]
    X1_test2 = X1[border2:border3]

    
    Xdata_train = []
    Xdata_test = []
    Xdata_test2 = []
    
    for i in Xdata:
        Xdata_train.append(i[:border1])
        Xdata_test.append(i[border1:border2])
        Xdata_test2.append(i[border2:border3])
    

    y_train,y_test, y_test2 = y[:border1],y[border1:border2], y[border2:border3]

    X1_train = X1_train.reshape((X1_train.shape[0],X1_train.shape[1],1))
    X1_test = X1_test.reshape((X1_test.shape[0],X1_test.shape[1],1))
    X1_test2 = X1_test2.reshape((X1_test2.shape[0],X1_test2.shape[1],1))

    Xdata_train_r = []
    Xdata_test_r = []
    Xdata_test2_r = []
    
    for i in range(newdim):
        Xdata_train_r.append(Xdata_train[i].reshape((Xdata_train[i].shape[0],Xdata_train[i].shape[1],1)))
        Xdata_test_r.append(Xdata_test[i].reshape((Xdata_test[i].shape[0],Xdata_test[i].shape[1],1)))
        Xdata_test2_r.append(Xdata_test2[i].reshape((Xdata_test2[i].shape[0],Xdata_test2[i].shape[1],1)))
    
    

    X_train_con = np.concatenate(Xdata_train_r, axis=2)
    X_test_con  = np.concatenate(Xdata_test_r, axis=2)
    X_test2_con  = np.concatenate(Xdata_test2_r, axis=2)
    

    X_train = [X1_train, X_train_con]
    X_test = [X1_test, X_test_con]
    X_test2 = [X1_test2, X_test2_con]


    input1 = Input(shape=(memory,1))
    input2 = Input(shape=(memory + predictions,newdim))



    x1 = GaussianNoise(inc_layer2, input_shape=(memory,1))(input1)
   
    x1 = GRU(units=inc_layer1, kernel_initializer = 'glorot_uniform', recurrent_initializer='orthogonal',
          bias_initializer='zeros', kernel_regularizer='l2', recurrent_regularizer=None,
          bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
          recurrent_constraint=None, bias_constraint=None, return_sequences=False,
          return_state=False, stateful=False)(x1)
    x1 = Dropout(drop1)(x1)
   
    x1 = Model(inputs=input1, outputs=x1)

    
    x2 = Dense(data_layer1, input_shape=(memory+predictions,3))(input2)
    x2 = Dropout(drop2)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(data_layer2)(x2)
    x2 = Model(inputs=input2, outputs=x2)

    combined = concatenate([x1.output, x2.output])

    z = Dense(dense_layer, activation="relu")(combined)
    z = Dense(predictions, activation="linear")(z)

    #define the model
    

    myadam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    


    class PlotResuls(Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []

            #self.fig = plt.figure()

            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.i += 1




            #print (".", end = '')
            if (epoch % 14999 == 0) & (epoch > 0):
                print(epoch)

                plt.plot(self.x, np.log(self.losses), label="loss")
                plt.plot(self.x, np.log(self.val_losses), label="val_loss")
                plt.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
                plt.title("Loss")
                plt.legend()
                plt.show();
                #mymanyplots(epoch, data, model)

    
    #data = [X1, X2, X3, X4, y, X1_train,X_train, X_test, X1_test, border1, border2, y_train, y_test, memory, y_temp, predictions]
    plot_results = PlotResuls()
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)
    ens_val_array = np.zeros(ensemble_count)
    ens_test_array = np.zeros(ensemble_count)
    
    for ens_no in range(ensemble_count):
        tf.keras.backend.clear_session()
        mc = ModelCheckpoint(f'best_model_ens_{ens_no}.h5', monitor='val_loss',
                             mode='min', save_best_only=True, verbose=0)
        model = Model(inputs=[x1.input, x2.input], outputs=z)
        model.compile(optimizer=myadam,loss='mean_squared_error')
        history = model.fit(X_train,y_train,validation_data=(X_test, y_test),
                            epochs=2000, verbose=0, batch_size=bs,
                            callbacks=[plot_results, es, mc])

        model = load_model(f'best_model_ens_{ens_no}.h5')
        valresult = np.log(model.evaluate(x= X_test, y=y_test, verbose=0))
        testresult = np.log(model.evaluate(x= X_test2, y=y_test2, verbose=0)) 
        
        ens_val_array[ens_no] = valresult
        ens_test_array[ens_no] = testresult

        
    winner = ens_val_array.argmin()
    model = load_model(f'best_model_ens_{winner}.h5')
    
    info(ens_val_array)
    info(ens_test_array)
    info(f'Validation winner {winner}')
# ======================================
    
#     plt.plot(np.log(history.history['loss']), label='loss')
#     plt.plot(np.log(history.history['val_loss']), label='test')
#     plt.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
#     plt.legend()
#     plt.clf() 
#     plt.show()
    
# =============================================================================
#     for i in range(5):
#          rand = np.random.randint(0,len(X_test[0]))
#          y_test_descaled = RNN_scaler.inverse_transform(y_test[rand, np.newaxis])
#          y_test_descaled = y_test_descaled.flatten()
# 
#          y_pred = model.predict(X_test)
#          
#          y_pred_descaled = RNN_scaler.inverse_transform(y_pred[rand, np.newaxis])
#          y_pred_descaled = y_pred_descaled.flatten()
#          
#          
#          plt.plot(y_test_descaled, label="true")
#          plt.plot(y_pred_descaled,label="predicted")
#          plt.title('Inclination delta')
#          #plt.ylim(0,1)
#          plt.legend()
#          plt.show()
# =============================================================================
# =============================================================================
#          
#          
# 
#          plt.plot(np.cumsum(y_test_descaled), label="true")
#          plt.plot(np.cumsum(y_pred_descaled),label="predicted")
#          plt.title('Inclination cumsum')
#          plt.legend()
#          plt.show()
# =============================================================================
    sample_count = len(X_test2[0])
    y_pred = model.predict(X_test2)
    
    #### Different ensemble, voting ######
    ypred_array = []
    for i in range(ensemble_count):
        model = load_model(f'best_model_ens_{i}.h5')
        y_pred = model.predict(X_test2)
        ypred_array.append(y_pred)

    y_pred = np.average(ypred_array, axis=0)
    
    
    ######## Different ensemble ends here #
    
    y_test_descaled = RNN_scaler.inverse_transform(y_test2)
    y_pred = model.predict(X_test2)
    y_pred_descaled = RNN_scaler.inverse_transform(y_pred)
 

    error_matrix = np.cumsum(y_pred_descaled, axis=1)- np.cumsum(y_test_descaled, axis=1)
    
    def rand_jitter(arr):
        stdev = .004*(max(arr)-min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, **kwargs):
        return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, verts=verts, **kwargs)

    
    
    plt.figure(figsize=(5,5), dpi=200)
    for i in range(sample_count):
        _ = jitter(np.arange(0,100,1),error_matrix[i], alpha=1,s=0.5,marker=".", c="black")
    plt.title(f"delta, drilled {percent_drilled}")
    #plt.plot(np.median(error_matrix, axis=0), linewidth=8, alpha=1, c="white")
    #plt.plot(np.median(error_matrix, axis=0), linewidth=2, alpha=1, c="black")
    plt.scatter(np.arange(0,100,1),np.average(np.abs(error_matrix), axis=0),
                marker="o",s=40, alpha=0.7, c="white", zorder=2)
    
    c_array = np.empty(100, dtype=object)
    aae = np.average(np.abs(error_matrix), axis=0)
    for i in range(100):
        if aae[i] <= 0.4:
            c_array[i] = "green"
        elif aae[i] <= 0.8:
            c_array[i] = "orange"
        else:
            c_array[i] = "red"
    
    plt.scatter(np.arange(0,100,1),aae, marker=".",s=20, alpha=1, c=c_array,
                zorder=3, label="Average Absolute Error")
    plt.ylim((-3,3))
    plt.axhline(y=0, xmin=0, xmax=1, linewidth=2, c="black")
    plt.axhline(y=0.4, xmin=0, xmax=1, linewidth=1, c="black")
    plt.axhline(y=0.8, xmin=0, xmax=1, linewidth=1, c="black")
    plt.legend()
    plt.show()     
  
         
         
    
    model = load_model('best_model.h5')
    #mymanyplots(-1, data, model)
    #myerrorplots(data, model)
    valresult = np.log(model.evaluate(x= X_test, y=y_test, verbose=0))
    testresult = np.log(model.evaluate(x= X_test2, y=y_test2, verbose=0))
   
    return valresult, testresult, aae