#!/usr/bin/env python
# coding: utf-8

# import module
import pandas as pd
import numpy as np
import sqlite3 as sql
import time

# coneect to sql
conn = sql.connect("simulated_data/jump_data.sqlite")
N_data = 100
for i in range(N_data):
    # import data
    file_name = 'data_' + str(i)
    data_raw = pd.DataFrame(conn.execute("SELECT * FROM  {} ".format(file_name)).fetchall())
    data_raw.to_csv('simulated_data/jump_data/'+file_name+'.csv')

