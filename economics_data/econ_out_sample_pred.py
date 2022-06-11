#!/usr/bin/env python
# coding: utf-8

# import module
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as backend
import sqlite3 as sql
import time
from sklearn.linear_model import LinearRegression
from lib.RNN_TVP import rescale_layer, linear_layer


# load data
data_raw = pd.read_csv('econ_data/three_variable.csv',header = None).dropna()
x_lag_1 = data_raw.loc[0:260,1:3].shift(1)
x_lag_2 = data_raw.loc[0:260,1:3].shift(2)
X = pd.concat([x_lag_1.loc[2:260,:],x_lag_2.loc[2:260,:]],axis=1)
X['intercept'] = np.ones([259,1])
y = data_raw.loc[2:260,1]

test_size = 120
training_size = y.shape[0] - test_size - 1
count = 0

y_predict = np.empty(0)
y_true = np.empty(0)
for i in range(test_size):
    # truncate training sample
    training_size = training_size+1
    
    x_train = X.iloc[0:(training_size - 1),:]
    y_train = y.iloc[0:(training_size - 1)]
    
    x_predict = X.iloc[training_size,:]
    y_true_temp = y.iloc[training_size]

    # transform data to tensor
    x_train = tf.cast(x_train, tf.float32)
    y_train = tf.cast(y_train, tf.float32)
    
    # initial value by linea regression
    reg = LinearRegression().fit(x_train, y_train)
    beta_pre_sample = reg.coef_

    # build the model
    data_input = keras.Input(shape=(x_train.shape[-2], x_train.shape[-1]))
    h_1 = keras.layers.SimpleRNN(16, return_sequences=True, name='RNN_1')(data_input)
    h_2 = keras.layers.LSTM(16, return_sequences=True, name='RNN_2')(h_1)
    beta_init = tf.constant_initializer(beta_pre_sample)
    h_fix = keras.layers.Dense(1, name='fix', kernel_initializer=beta_init,
                               kernel_regularizer=keras.regularizers.l1(0.1))(data_input)
    tvp = rescale_layer(16, x_train, x_train.shape[-1], name='rescale', regularizers = keras.regularizers.l1(0.1))(h_2)
    linear = linear_layer(3, x_train, name='linear')(tvp, h_fix)
    model = keras.Model(inputs=data_input, outputs=linear)
    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    # model fit
    model.fit(x=x_train[tf.newaxis, :, :], y=y_train[tf.newaxis, :, tf.newaxis], epochs=2000, verbose=0)
    
    # beta predict
    layer = model.get_layer('rescale')
    extractor = keras.Model(inputs=model.inputs,
                    outputs=layer.output)
    features = np.array(extractor(x_train[tf.newaxis, :, :]))[0, :, :]
    fix_layer = model.get_layer('fix')
    beta_fix = np.array(fix_layer.get_weights())[0]
    beta_final = features + np.transpose(beta_fix)
    beta_predict = beta_final[-1,:]
    
    y_predict = np.append(y_predict,np.sum(np.multiply(x_predict, beta_predict)))
    y_true =  np.append(y_true,y_true_temp)
    
    count = count + 1
    print(str(100*count/test_size),'%')


# save result table
result = pd.concat([pd.DataFrame(y_predict),pd.DataFrame(y_true)],axis=1)
result.to_csv('result/out_sample_pred_RNN.csv')

