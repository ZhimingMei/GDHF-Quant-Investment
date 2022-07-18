import pandas_datareader.data as web
import datetime as dt

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2021, 9, 1)
# datareader 读取股票信息
df = web.DataReader('GOOGL', 'stooq', start, end)
#删除空值
df.dropna(inplace=True)
# 对时间进行排序
df.sort_index(inplace=True)
# 收盘价上移10天，以10天为一组训练预测值
pre = 10
df['label'] = df['Close'].shift(-pre)
df.to_csv('google_stock', sep=',')