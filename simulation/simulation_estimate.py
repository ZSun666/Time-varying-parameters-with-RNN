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

# setup
N_data = 100  # number of data set

# connect to the sqlite database
conn = sql.connect("simulated_date/smooth_data.sqlite")
conn_1 = sql.connect("simulated_date/smooth_result.sqlite")

# iterate over all the tables
start = time.time()
for i in range(N_data):
    # import data
    file_name = 'data_' + str(i)
    data_raw = pd.DataFrame(conn.execute("SELECT * FROM  {} ".format(file_name)).fetchall())
    x_train = data_raw.iloc[:, 4:7]
    y_train = data_raw.iloc[:, 7]
    x_train = tf.cast(x_train, tf.float32)
    y_train = tf.cast(y_train, tf.float32)

    # initilize the beta by OLS
    reg = LinearRegression().fit(x_train, y_train)
    beta_pre_sample = reg.coef_

    # build the model
    data_input = keras.Input(shape=(x_train.shape[-2], x_train.shape[-1]))
    h_1 = keras.layers.SimpleRNN(16, return_sequences=True, name='RNN_1')(data_input)
    h_2 = keras.layers.LSTM(16, return_sequences=True, name='RNN_2')(h_1)
    beta_init = tf.constant_initializer(beta_pre_sample)
    h_fix = keras.layers.Dense(1, name='fix', kernel_initializer=beta_init,
                               kernel_regularizer=keras.regularizers.l1(0.01))(data_input)
    tvp = rescale_layer(16, x_train, x_train.shape[-1], name='rescale')(h_2)
    linear = linear_layer(3, x_train, name='linear')(tvp, h_fix)
    model = keras.Model(inputs=data_input, outputs=linear)
    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    # model fit
    model.fit(x=x_train[tf.newaxis, :, :], y=y_train[tf.newaxis, :, tf.newaxis], epochs=1000, verbose=0)

    # save coefficients
    layer = model.get_layer('rescale')
    extractor = keras.Model(inputs=model.inputs,
                            outputs=layer.output)
    features = np.array(extractor(x_train[tf.newaxis, :, :]))[0, :, :]
    fix_layer = model.get_layer('fix')
    beta_fix = np.array(fix_layer.get_weights())[0]
    beta_final = features + np.transpose(beta_fix)

    # save fitted y
    layer_1 = model.get_layer('linear')
    extractor = keras.Model(inputs=model.inputs,
                            outputs=layer_1.output)
    y_hat = np.array(extractor(x_train[tf.newaxis, :, :]))[0, :]

    # save the result into database
    save_table = pd.DataFrame(beta_final)
    save_table.insert(0, 'fitted_y', y_hat)
    save_table.insert(0, 'y_true', y_train)
    save_table.to_sql(file_name, con=conn_1)
    end = time.time()
    print("time:" + str((end - start) / 60) + "min")
    print(str(i) + "/" + str(N_data))
