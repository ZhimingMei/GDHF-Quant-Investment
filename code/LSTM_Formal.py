import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from collections import deque
from keras.models import load_model

# 读取股票数据与因子数据
data = pd.read_csv('/Users/ryan/Documents/GitHub/GDHF-Quant-Investment/data/data_cleaned/our_factor_cleaned.gz')

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

# 导入最优模型
best_model = load_model('/Users/ryan/Documents/GitHub/GDHF-Quant-Investment/5.09_46mem_10_lstm_2_dense_2_unit_32')

# 查看模型信息
#best_model.summary()

# 效果检验
x, y = LSTM_stock(stock, 10)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, shuffle=False, test_size=0.1)
best_model.evaluate(x_te, y_te)

# 画图
df_time = stock.index[-len(y_te):]
# 实际股票表现
plt.plot(df_time, y_te, color='red', label='price')
# 预测股票表现
pre = best_model.predict(x_te)

# Datetime 转化
Str_date = []
for i in stock['TRADE_DT'][-len(y_te):]:
    Str_date.append(str(i))
dates = []
for s in Str_date:
    d  = pd.to_datetime(s).date()
    dates.append(d)

plt.plot(df_time, pre, color='green', label='pridict')
#plt.xticks(df_time, dates)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(26))
plt.legend()
plt.show()