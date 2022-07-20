import pandas as pd

# 中位数去极值法 并且标准化
def filter_MAD(df,factor,n=5):
    # 去极值
    median = df[factor].quantile(0.5)
    new_median =((df [factor]- median).abs()).quantile(0.5)
    max_range = median + n * new_median
    min_range = median - n * new_median
    for i in range(df.shape[0]):
        if df.loc[i, factor] > max_range:
            df.loc[i, factor] = max_range
        elif df.loc[i, factor] < min_range:
            df.loc[i,factor] = min_range

    #标准化
    df[factor] = (df[factor] - df[factor].mean()) / df[factor].std()

    return df

cf = pd.read_table('D:/Desktop/厚方 投资建模/value_factor.txt',sep=',')
cf = cf.sort_values(by = 'TRADE_DT')

grouped = cf.groupby('TRADE_DT')
factor_list = ['S_VAL_PE','S_VAL_PB_NEW','S_VAL_PS','S_DQ_TURN','S_DQ_MV','NET_ASSETS_TODAY']
L=[]
for i in grouped:
    a = i[1]
    a = a.reset_index(drop=True)
    a = a.fillna(a.mean()) # 以日频数据为单位，已均值填充为缺失值
    for factor in factor_list:
        a = filter_MAD(a, factor)
    L.append(a)

# 数据合并  
b = L[0]
for i in L[1:]:
    b = b.append(i)
b = b.reset_index(drop=True)

print(b)
b.to_csv('value_factor_cleaned.gz', compression='gzip', index=False)
