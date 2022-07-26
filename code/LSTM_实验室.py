import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import deque
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# 读取股票数据与因子数据
data = pd.read_csv('merged_daily_freq_factor_cleaned')

# 单一股票 Test
stock = data.groupby('S_INFO_WINDCODE').get_group('600111.SH')

# 封装函数，输入值为数据表、记忆天数
def LSTM_stock(stock, mem_day):

    # 创建 FIFO 队列
    deq = deque(maxlen=mem_day)

    # 建立自变量
    x = []
    for i in range(len(stock)):
        deq.append(list(stock.iloc[i, 9:15]))
        if len(deq) == mem_day:
            x.append(list(deq))
    x = np.array(x)

    # 创建输出集 y
    y = stock['S_DQ_CLOSE'][mem_day - 1:].values

    # 函数输出输入值 list、输出值、NaN 值行
    return x, y

mem_days = [5, 10, 15]
lstm_layers = [1, 2, 3]
dense_layers = [1, 2, 3]
units = [16, 32]

# 模型优化
def opt_model(mem_days, lstm_layers, dense_layers, units):
    for m in mem_days:
        for l in lstm_layers:
            for d in dense_layers:
                for u in units:
                    file_path = 'F:/文档/All_codes/models/{val_mape:.2f}_{epoch:02d}' + \
                        f'mem_{m}_lstm_{l}_dense_{d}_unit_{u}'
                    cp = ModelCheckpoint(
                        filepath=file_path,
                        save_weights_only=False,
                        monitor='val_mape',
                        mode='min',
                        save_best_only=True
                    )
                    # 调用函数
                    x, y= LSTM_stock(stock, m)
                    # 分割训练集与测试集（shuffle=False 使顺序不乱）
                    x_tr, x_te, y_tr, y_te = train_test_split(x, y, shuffle=False, test_size=0.1)

                    # 构建神经网络
                    model = tf.keras.Sequential()
                    # 构建第一层神经网络
                    model.add(LSTM(u, input_shape=x.shape[1:], activation='relu', \
                        return_sequences=True))
                    # 防止过拟合
                    model.add(Dropout(0.1))

                    for ls in range(l):
                        # 构建第二层神经网络
                        model.add(LSTM(u, activation='relu', return_sequences=True))
                        model.add(Dropout(0.1))

                    # 构建第三层神经网络
                    model.add(LSTM(u, activation='relu'))
                    model.add(Dropout(0.1))

                    for ds in range(d):
                        # 构建全连接层
                        model.add(Dense(u, activation='relu'))
                        model.add(Dropout(0.1))

                    # 构建输出层
                    model.add(Dense(1))

                    # 编译（优化器、损失函数、评价函数）
                    model.compile(optimizer='adam', loss='mse', metrics=['mape'])

                    #训练（该模型偏差在 13% 左右）
                    model.fit(x_tr, y_tr, batch_size=32, epochs=50, validation_data=(x_te, y_te), \
                        callbacks=[cp])

# 跑了两个小时的模型测试……
opt_model(mem_days, lstm_layers, dense_layers, units)