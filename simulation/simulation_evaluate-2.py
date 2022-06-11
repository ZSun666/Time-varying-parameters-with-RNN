#!/usr/bin/env python
# coding: utf-8


# import module
import pandas as pd
import numpy as np
import sqlite3 as sql
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# function: cumulative sum
def cum_sum(lists):
    clist = []
    length = len(lists)
    clist = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return clist[1:]


DGP_type = 1 # 1 is smooth transition; 2 is jump transition


N_data = 100
if DGP_type == 1:
    conn = sql.connect("simulated_data/smooth_result.sqlite")
else:
    conn = sql.connect("simulated_data/jump_result.sqlite")
    
result_RNN = np.empty([200,6,100])
result_RW = np.empty([200,4,100])

# forloop over all the simulated tables
for i in range(N_data):
    # import results by RNN
    file_name = 'data_' + str(i)
    data_temp = np.array(conn.execute("SELECT * FROM  {} ".format(file_name)).fetchall())
    result_RNN[:,:,i] = data_temp
    # import results by RW
    if DGP_type == 1:
        file_name_RW = 'simulated_data/smooth_result/result_' + str(i)+'.csv'
    else:
        file_name_RW = 'simulated_data/jump_result/result_' + str(i)+'.csv'
    data_temp = pd.read_table(file_name_RW,sep = ',',header = None)
    result_RW[:,:,i] = data_temp.iloc[:,4:8]
    
# take the mean of all the simulated data
result_RNN_mean = np.mean(result_RNN,2)
result_RW_mean = np.mean(result_RW,2)
beta_RNN_mean = result_RNN_mean[:,3:6]
beta_RW_mean = result_RW_mean[:,1:4];

# connect to sql 
if DGP_type == 1:
    conn = sql.connect("simulated_data/smooth_data.sqlite")
else:
    conn = sql.connect("simulated_data/jump_data.sqlite")
    
data_temp = np.array(conn.execute("SELECT * FROM  {} ".format('data_0')).fetchall())
beta_true = data_temp[:,1:4]





# Table evaluate fitted beta and y
MSE = pd.DataFrame();
for i in range(3):
    MSE.loc[1,i] = np.square((beta_true[:,i] - beta_RNN_mean[:,i])).mean(0)
    MSE.loc[2,i] = np.square((beta_true[:,i] - beta_RW_mean[:,i])).mean(0)
MSE.loc[1,i+1] = np.square(result_RNN_mean[:,1] - result_RNN_mean[:,2]).mean(0)
MSE.loc[2,i+1] = np.square(result_RW_mean[:,0] - result_RNN_mean[:,2]).mean(0)

CSE_RNN = np.zeros([200,3])
CSE_RW = np.zeros([200,3])
# Cumulative Mean Squared Error
CSE_RNN = np.array(cum_sum(np.square(beta_true - beta_RNN_mean)))
CSE_RW = np.array(cum_sum(np.square(beta_true - beta_RW_mean)))
CSE_diff = CSE_RNN - CSE_RW

CSE_RNN_y = np.array(cum_sum(np.square(result_RNN_mean[:,1] - result_RNN_mean[:,2])))
CSE_RW_y =  np.array(cum_sum(np.square(result_RW_mean[:,0] - result_RNN_mean[:,2])))
CSE_diff_y = CSE_RNN_y - CSE_RW_y


# Figure fitted beta
x = range(200)
fig = plt.figure(figsize=(48, 24))
grid = plt.GridSpec(3,6, hspace=0.2, wspace=0.2)
main_beta_1 = fig.add_subplot(grid[:-1, 0:2])
CSE_beta_1 = fig.add_subplot(grid[-1,0:2],sharex = main_beta_1)
main_beta_2 = fig.add_subplot(grid[:-1, 2:4])
CSE_beta_2 = fig.add_subplot(grid[-1,2:4], sharex = main_beta_2)
main_beta_3 = fig.add_subplot(grid[:-1, 4:6])
CSE_beta_3 = fig.add_subplot(grid[-1,4:6],sharex = main_beta_3)

main_beta_1.plot(x, beta_true[:,0],label='True value')
main_beta_1.plot(x, beta_RNN_mean[:,0],label='RNN-TVP')
main_beta_1.plot(x, beta_RW_mean[:,0],label='RW-TVP')
CSE_beta_1.fill_between(x,CSE_diff[:,0],label = 'CSE')
main_beta_1.legend()
CSE_beta_1.legend()

main_beta_2.plot(x, beta_true[:,1],label='True value')
main_beta_2.plot(x, beta_RNN_mean[:,1],label='RNN-TVP')
main_beta_2.plot(x, beta_RW_mean[:,1],label='RW-TVP')
CSE_beta_2.fill_between(x,CSE_diff[:,1],label = 'CSE')
main_beta_2.legend()
CSE_beta_2.legend()

main_beta_3.plot(x, beta_true[:,2],label='True value')
main_beta_3.plot(x, beta_RNN_mean[:,2],label='RNN-TVP')
main_beta_3.plot(x, beta_RW_mean[:,2],label='RW-TVP')
main_beta_3.set_ylim([-0.5,0.5])
CSE_beta_3.fill_between(x,CSE_diff[:,2],label = 'CSE')
CSE_beta_3.set_ylim([-0.2,0.2])
main_beta_3.legend()
CSE_beta_3.legend()


# Save the fig
if DGP_type == 1:
    plt.savefig('figure/simulation_smooth_beta')
else:
    plt.savefig('figure/simulation_jump_beta')


# Figure fitted y
x = range(200)

# Save the fig


fig = plt.figure(figsize=(9, 9))
grid = plt.GridSpec(3, 2, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 0:])
mse_ax = fig.add_subplot(grid[-1,0:],sharex=main_ax)


main_ax.plot(x, result_RNN_mean[:,1],label='True value')
main_ax.plot(x, result_RNN_mean[:,2],label='RNN-TVP')
main_ax.plot(x, result_RW_mean[:,0],label='RW-TVP')
main_ax.legend()
mse_ax.fill_between(x, CSE_diff_y,label = 'CSE')
mse_ax.legend()

if DGP_type == 1:
    plt.savefig('figure/simulation_smooth_fit')
else:
    plt.savefig('figure/simulation_jump_fit')

