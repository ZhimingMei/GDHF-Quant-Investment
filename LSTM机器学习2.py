import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import deque
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout

df = pd.read_csv('google_stock')
scaler = StandardScaler()
# 标准化
sca_x = scaler.fit_transform(df.iloc[:, 1:-1])

# 设定记忆天数
mem = 10
# 创建 FIFO 队列
deq = deque(maxlen=mem)

x = []
for i in sca_x:
    deq.append(list(i))
    if len(deq) == mem:
        x.append(list(deq))

# 创建输入集 x，删除 label=NaN 的行
x_latest = x[-10:]
x = np.array(x[:-10])

# 创建输出集 y
y = df['label'][mem - 1:-10].values
# 分割训练集与测试集
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.1)


# 构建神经网络
model = tf.keras.Sequential()
# 构建第一层神经网络
model.add(LSTM(10, input_shape=x.shape[1:], activation='relu', return_sequences=True))
# 防止过拟合
model.add(Dropout(0.1))

# 构建第二层神经网络
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(Dropout(0.1))

# 构建第三层神经网络
model.add(LSTM(10, activation='relu'))
model.add(Dropout(0.1))

# 构建全连接层
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))

# 构建输出层
model.add(Dense(1))

# 编译（优化器、损失函数、评价函数）
model.compile(optimizer='adam', loss='mse', metrics=['mape'])

#训练（该模型偏差在 13% 左右）
model.fit(x_tr, y_tr, batch_size=32, epochs=50, validation_data=(x_te, y_te))