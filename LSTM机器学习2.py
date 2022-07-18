import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import deque
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt

df = pd.read_csv('google_stock')

# 封装函数，输出值为数据表、记忆天数、预测天数
def LSTM_stock(sheet, mem_day, pre_day):
    scaler = StandardScaler()
    # 标准化
    sca_x = scaler.fit_transform(sheet.iloc[:, 1:-1])

    # 创建 FIFO 队列
    deq = deque(maxlen=mem_day)

    x = []
    for i in sca_x:
        deq.append(list(i))
        if len(deq) == mem_day:
            x.append(list(deq))

    # 创建输入集 x，删除 label=NaN 的行
    x_latest = x[-pre_day:]
    x = np.array(x[:-pre_day])
    # 创建输出集 y
    y = df['label'][mem_day - 1:-pre_day].values

    # 函数输出输入值 list、输出值、NaN 值行
    return x, y, x_latest

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
                        f'mem_{m}_lstm{l}_dense_{d}_unit_{u}'
                    cp = ModelCheckpoint(
                        filepath=file_path,
                        save_weights_only=False,
                        monitor='val_mape',
                        mode='min',
                        save_best_only=True
                    )
                    # 调用函数
                    x, y, x_latest = LSTM_stock(df, m, 10)
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

# 导入最优模型（本次测试最优误差率为 4.87%）
best_model = load_model('F:/文档/All_codes/models/\
4.87_29mem_[5, 10, 15]_lstm[1, 2, 3]_dense_[1, 2, 3]_unit_[16, 32]')

# 查看模型信息
best_model.summary()

# 效果检验
x, y, x_latest = LSTM_stock(df, 5, 10)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, shuffle=False, test_size=0.1)
best_model.evaluate(x_te, y_te)

# 画图
df_time = df.index[-len(y_te):]
# 实际股票表现
plt.plot(df_time, y_te, color='red', label='price')
# 预测股票表现
pre = best_model.predict(x_te)
plt.plot(df_time, pre, color='green', label='pridict')
plt.legend()
plt.show()