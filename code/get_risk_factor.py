"""
author: zhimingmei
date: 2022/07/21
description: 通过股票日行情数据获得风险类因子数据
"""


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

path = '/Users/ryan/Documents/GitHub/GDHF-Quant-Investment/data'
file_name = 'eod_price.gz'
input_file = os.path.join(path, file_name)
df = pd.read_csv(input_file)

# Variance20
df.set_index('TRADE_DT', inplace=True)
df.sort_index(inplace=True)
var20 = df[['S_INFO_WINDCODE','S_DQ_PCTCHANGE']].groupby('S_INFO_WINDCODE').apply(lambda x:
     x['S_DQ_PCTCHANGE'].rolling(20).var())
var20 = pd.DataFrame(var20)
var20.columns = ['Variance20']
var20.reset_index(inplace=True)

# Kurtosis20
kurt20 = df[['S_INFO_WINDCODE','S_DQ_PCTCHANGE']].groupby('S_INFO_WINDCODE').apply(lambda x:
     x['S_DQ_PCTCHANGE'].rolling(20).apply(lambda y: y.kurtosis()))
kurt20 = pd.DataFrame(kurt20)
kurt20.columns = ['Kurtosis20']
kurt20.reset_index(inplace=True)

# Skewness20
skew20 = df[['S_INFO_WINDCODE','S_DQ_PCTCHANGE']].groupby('S_INFO_WINDCODE').apply(lambda x:
     x['S_DQ_PCTCHANGE'].rolling(20).apply(lambda y: y.skew()))
skew20 = pd.DataFrame(skew20)
skew20.columns = ['Skewness20']
skew20.reset_index(inplace=True)

# SharpeRatio20
sharpe20 = df[['S_INFO_WINDCODE','S_DQ_PCTCHANGE']].groupby('S_INFO_WINDCODE').apply(lambda x:
     x['S_DQ_PCTCHANGE'].rolling(20).apply(lambda y: (y.mean()-0.04/252*20)/math.sqrt(y.var())))
sharpe20 = pd.DataFrame(sharpe20)
sharpe20.columns = ['SharpeRatio20']
sharpe20.reset_index(inplace=True)

# merge four dataframes
data1 = pd.merge(var20, skew20, on=['TRADE_DT','S_INFO_WINDCODE'],how='left')
data2 = pd.merge(kurt20, sharpe20, on=['TRADE_DT','S_INFO_WINDCODE'],how='left')
data = pd.merge(data1, data2, on=['TRADE_DT','S_INFO_WINDCODE'],how='left')

# final dataset
file_name = 'risk_factor.gz'
data.to_csv(os.path.join(path, file_name), compression='gzip', index=False)