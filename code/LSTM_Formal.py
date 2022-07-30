import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
# 导入 LSTM_stock 函数（两份代码文件需在同一文件夹）
from LSTM_Test import LSTM_stock

# 读取股票数据与因子数据
data = pd.read_csv('new_factor_data')

# 导入检验股票
stock = data.groupby('S_INFO_WINDCODE').get_group('600111.SH')

# 导入最优模型
best_model = load_model('F:/文档/All_codes/models/6.00_45mem_5_lstm_2_dense_3_unit_32')

# 查看模型信息
best_model.summary()

# 效果检验
x, y = LSTM_stock(stock, 5)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, shuffle=False, test_size=0.1)
best_model.evaluate(x_te, y_te)

# 画图
df_time = stock.index[-len(y_te):]
# 实际股票表现
plt.plot(df_time, y_te, color='red', label='price')

# 预测股票表现
pre = best_model.predict(x_te)
# 预测股价变化
plt.plot(df_time, pre, color='green', label='predict')

plt.legend()
plt.show()