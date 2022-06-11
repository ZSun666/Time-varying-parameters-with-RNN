#!/usr/bin/env python
# coding: utf-8

# import modules
import pandas as pd
import numpy as np
import sqlite3
import time
import datetime 
import pandas as pd
import numpy as np
import math
import statistics as stat
import matplotlib.pyplot as plt
import sqlite3
import sys
sys.path.insert(0, './lib')
from simulate_data import simulate_data

# length of simulated data
T = 200
simulate = simulate_data(T)

# connect to the sql 
conn = sqlite3.connect('simulated_data/jump.sqlite')

# num of simulated data set
n_num = 100

# generate simulations and save
for i in range(n_num):
    table_name = "data_" + str(i)
    [beta,x,y] = simulate.simulate_jump()
    data = pd.DataFrame(np.column_stack([beta,x,y]))
    data.to_sql(table_name,con = conn)

