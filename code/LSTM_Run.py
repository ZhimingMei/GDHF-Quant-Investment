import pandas as pd
import numpy as np
from collections import deque
from keras.models import load_model

# 循环打表
def running(sheet, mod):
    stocks = sheet.groupby('S_INFO_WINDCODE')
    st_name = sheet['S_INFO_WINDCODE'].unique()
    for stk in st_name:
        stock = stocks.get_group(stk)
        x, y = LSTM_stock(stock, 5)
        pre = mod.predict(x)
        form_table(stk, stock['TRADE_DT'][4:], pre, y)

# 封装函数，输入值为数据表、记忆天数
def LSTM_stock(stock, mem_day):

    # 创建 FIFO 队列
    deq = deque(maxlen=mem_day)

    # 建立自变量
    x = []
    for i in range(len(stock)):
        deq.append(list(stock.iloc[i, 3:]))
        if len(deq) == mem_day:
            x.append(list(deq))
    x = np.array(x)

    # 创建输出集 y
    y = stock['S_DQ_CLOSE'][mem_day - 1:].values

    # 函数输出输入值 list、输出值、NaN 值行
    return x, y

# 输出表格
def form_table(name, date, pre, pri):
    f, s = name_cut(name)
    pre_tableu = []
    for i in range(len(pre)):
        layer = [date.iloc[i], pre[i][0], pri[i]]
        pre_tableu.append(layer)
    tableu = pd.DataFrame(pre_tableu, columns=['TRADE_DT', 'PREDICT_PR', 'TRUE_PR'])
    tableu.to_csv('F:/文档/All_codes/tableus/%s_%s.gz' %(f, s), compression='gzip', index=False)

# 文件名处理
def name_cut(name):
    str_list = name.split('.')
    fir, sec = str_list[0], str_list[1]
    return fir, sec

def main():
    # 读取股票数据与因子数据
    data = pd.read_csv('new_factor_data')
    # 导入最优模型
    best_model = load_model('F:/文档/All_codes/models/6.00_45mem_5_lstm_2_dense_3_unit_32')
    running(data, best_model)

main()