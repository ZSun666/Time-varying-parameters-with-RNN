#!/usr/bin/env python
# coding: utf-8

# import module
import pandas as pd
import numpy as np
import sqlite3 as sql
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# load result
result_RNN = pd.read_csv('result/in_sample_fit')
#result_RNN.iloc[:,3:9]
beta_RNN = result_RNN.iloc[:,3:9]
y_true = result_RNN['y_true']
y_fit = result_RNN['fitted_y']


# Figure fitted beta
x = range(259)
fig, (axs_1,axs_2) = plt.subplots(2,3,sharex=True,figsize=(15,15))
for i in range(3):
    #axs[i].plot(x, beta_true[:,i],label='True value')
    axs_1[i].plot(x, beta_RNN.iloc[:,(i-1)*2+1],label='RNN-TVP')
    axs_2[i].plot(x, beta_RNN.iloc[:,(i-1)*2+2],label='RNN-TVP')

    #axs[i].plot(x, beta_MCMC_mean[:,i],label='RW-TVP')
axs[0].legend()

# Save the fig
#plt.savefig('figure/simulation_smooth_beta')




# Figure fitted y
x = range(259)
plt.plot(x,y_true,label='True value')
plt.plot(x,y_fit,label='RNN-TVP')
#plt.plot(x,result_MCMC_mean[:,0],label='RW-TVP')
plt.legend()

