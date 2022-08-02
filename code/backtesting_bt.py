'''

author: 28zhimingmei
date: 07/28/2022
description: 根据预测数据进行多股回测, 并对投资组合进行分析

'''


import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt

import os
import warnings

warnings.simplefilter('ignore')

path = '/Users/ryan/Documents/GitHub/GDHF-Quant-Investment/test_data'
file_name = 'test_data_all.parquet'

data = pd.read_parquet(os.path.join(path, file_name))

class GetKdatas(object):
    def __init__(self, secu_lst, benchmark='399300'):
        """
        :parameter secu_lst: a dict contained stocks with starts and ends
        :parameter benchmark: the name of benchmark
        """
        self.secu_lst = secu_lst
        self.benchmark = benchmark

    # 需要更改开始和结束时间
    @staticmethod
    def get_single_kdata(code, start='2020-01-01', end='2021-08-01', index=False):
        df = data[data['code']==code]
        df = df[df.datetime.between(start, end)]
        # df['datetime'] = pd.to_datetime(df['datetime'])
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'return_predict']]

    def get_all_kdata(self):
        kdata = {}
        for secu in set(self.secu_lst):
            secu_kdata = self.get_single_kdata(secu, self.secu_lst[secu]['start'], self.secu_lst[secu]['end'])
            kdata[secu] = secu_kdata.reset_index(drop=True)
        return kdata

    def merge_period(self):
        all_kdata = self.get_all_kdata()
        benchmark_start = min(self.secu_lst.values(), key=lambda x: x['start'])['start']
        benchmark_end = max(self.secu_lst.values(), key=lambda x: x['end'])['end']
        all_kdata['benchmark'] = self.get_single_kdata(self.benchmark, benchmark_start, benchmark_end,True)

        for secu in set(all_kdata.keys()) - set(['benchmark']):
            secu_kdata = all_kdata['benchmark'][['datetime']].merge(all_kdata[secu], how='left')
            secu_kdata['suspend'] = 0
            secu_kdata.loc[secu_kdata['open'].isnull(), 'suspend'] = 1  # 标记为停盘日
            secu_kdata.set_index(['datetime'], inplace=True)  # 设date为index
            end = secu_lst[secu]['end']
            secu_kdata.fillna(method='ffill', inplace=True)  # start后的数据用前日数据进行补充
            secu_kdata.fillna(value=0, inplace=True)  # start前的数据用0补充
            secu_kdata.loc[(secu_kdata.index > end), 'suspend'] = 1
            all_kdata[secu] = secu_kdata

        _ = all_kdata.pop('benchmark')
        return all_kdata

class CommInfoPro(bt.CommInfoBase):

    params = (
        ('stamp_duty', 0.001),  # 印花税率
        ('stamp_duty_fe', 1),  # 最低印花税
        ('commission', 0.001),  # 佣金率
        ('commission_fee', 5),  # 最低佣金费
        ('stocklike', True), # 股票
        ('commtype', bt.CommInfoBase.COMM_PERC), # 按比例收
    )

    def _getcommission(self, size, price, pseudoexec):
        '''
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it idicates a short / selling of shares.
        '''

        if size > 0:  # 买入，不考虑印花税
            return max(size * price * self.p.commission, self.p.commission_fee)
        elif size < 0:  # 卖出，考虑印花税
            return max(size * price * (self.p.stamp_duty + self.p.commission), self.p.stamp_duty_fe)
        else:
            return 0  # just in case for some reason the size is 0.

class PandasData(bt.feeds.DataBase):
    params = (
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5)
    )

class PandasData_Extend(bt.feeds.PandasData):
    lines = ('return_predict',)
    params = (
        ('return_predict', -1),
    )

class CommInfoPro(bt.CommInfoBase):

    params = (
        ('stamp_duty', 0.001),  # 印花税率
        ('stamp_duty_fe', 1),  # 最低印花税
        ('commission', 0.001),  # 佣金率
        ('commission_fee', 5),  # 最低佣金费
        ('stocklike', True), # 股票
        ('commtype', bt.CommInfoBase.COMM_PERC), # 按比例收
    )

    def _getcommission(self, size, price, pseudoexec):
        '''
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it idicates a short / selling of shares.
        '''

        if size > 0:  # 买入，不考虑印花税
            return max(size * price * self.p.commission, self.p.commission_fee)
        elif size < 0:  # 卖出，考虑印花税
            return max(size * price * (self.p.stamp_duty + self.p.commission), self.p.stamp_duty_fe)
        else:
            return 0  # just in case for some reason the size is 0.

class PandasData(bt.feeds.DataBase):
    params = (
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5)
    )

class PandasData_Extend(bt.feeds.PandasData):
    lines = ('return_predict',)
    params = (
        ('return_predict', -1),
    )

class TestStrategy(bt.Strategy):

    def __init__(self):
        self.order = None
        self.buy_lst = []

    def log(self, txt, dt=None):
        '''输出日志'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    
    def prenext(self):
        '''检验数据是否对齐'''
        pass
    
    def downcast(self, amount, lot):
        return abs(amount//lot*lot)
    
    def next(self):
        if self.order:
            return
        
        if self.datas[0].datetime.date(0) == end_date:
            return
        
        self.log(f'{self.broker.getvalue():.2f}, {[(x, self.getpositionbyname(x).size) for x in self.buy_lst]}')
        if not self.position:    # Done
            for secu in set(self.getdatanames()) - set(self.buy_lst):
                data = self.getdatabyname(secu)
                # 设置买入条件
                if (data.return_predict > 0):
                    # 买入价格为当前资金的多少倍
                    order_value = self.broker.getvalue()*0.01
                    order_amount = self.downcast(order_value/data.close[0], 100)
                    self.order = self.buy(data, size=order_amount, name=secu)
                    self.log(f'Buy:{secu}, price:{data.close[0]:.2f}, amount:{order_amount}')
                    self.buy_lst.append(secu)
                    print(self.buy_lst)
        
        else:
            now_list = []
            for secu in self.buy_lst:
                data = self.getdatabyname(secu)
                # 设置卖出条件
                if (data.return_predict < 0):
                    self.order = self.order_target_percent(data, 0, name=secu)
                    # self.order = self.sell(data)
                    # self.order = self.sell(data, size = self.getposition(data).size)
                    self.log(f"Sell: {secu}, price:{data.close[0]:.2f}, pct: 0")
                    continue
                now_list.append(secu)
            self.buy_lst = now_list.copy()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(f"""Buy:{order.info['name']}, amount:{order.executed.size}, price:{order.executed.price:.2f}""")
            elif order.issell():
                self.log(f"""Sell:{order.info['name']}, amount:{order.executed.size}, price:{order.executed.price:.2f}""")
            self.bar_executed = len(self)

        # Write down: no pending order
        self.order = None

# 填充需要回测的股票池
secu_lst = {'600111': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600383': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600837': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600048': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601138': {'start': '2018-07-12', 'end': '2022-03-31'}, 
'600918': {'start': '2020-07-08', 'end': '2022-03-31'}, 
'000800': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002736': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601238': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600655': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600352': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601808': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600958': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600188': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600489': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002493': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600000': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601788': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000776': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002202': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601698': {'start': '2019-07-31', 'end': '2022-03-31'}, 
'601166': {'start': '2017-02-10', 'end': '2022-03-31'},
'000703': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601688': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000001': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600426': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000877': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600999': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601838': {'start': '2018-03-12', 'end': '2022-03-31'}, 
'002568': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600926': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601211': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601236': {'start': '2019-08-07', 'end': '2022-03-31'}, 
'600176': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000625': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601878': {'start': '2017-07-27', 'end': '2022-03-31'}, 
'001979': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601881': {'start': '2017-03-02', 'end': '2022-03-31'}, 
'600900': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600460': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002050': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600183': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000338': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601088': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600886': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002648': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000708': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600089': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601186': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000066': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'603392': {'start': '2020-06-04', 'end': '2022-03-31'}, 
'002236': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000568': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600346': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600030': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601800': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'300207': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600276': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600690': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002129': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002602': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601111': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600362': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601633': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002709': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601066': {'start': '2018-07-23', 'end': '2022-03-31'}, 
'300628': {'start': '2017-04-21', 'end': '2022-03-31'}, 
'300274': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600809': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000876': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000895': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601229': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600438': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'300316': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'603185': {'start': '2019-02-01', 'end': '2022-03-31'}, 
'300347': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'300033': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601012': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600436': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002142': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'603899': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'601100': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600150': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002601': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600887': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002410': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002049': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'300014': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002074': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'300496': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002241': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'300433': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002414': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002475': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'300124': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000786': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'600406': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'000768': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002120': {'start': '2017-02-10', 'end': '2022-03-31'}, 
'002001': {'start': '2017-02-10', 'end': '2022-03-31'}
            }

kdata = GetKdatas(secu_lst).merge_period()
kdata = dict(sorted(kdata.items()))
    
# 开始回测
cerebro = bt.Cerebro()
cerebro.addstrategy(TestStrategy)

for secu in kdata.keys():
    df = kdata[secu]
    
    data = PandasData_Extend(dataname=df, 
    fromdate=df.index[0], 
    todate=df.index[-1]
    )

    cerebro.adddata(data, name=secu)

end_date = df.index[-1]


# 设置初始资本为1 million
startcash = 10**6
cerebro.broker.setcash(startcash)
print(f"初始资金{cerebro.broker.getvalue()}")
# 设置交易手续费
cerebro.broker.addcommissioninfo(CommInfoPro())

# 加入指标
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_sharpe')
cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_annrtn')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_dd')
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='_pyfolio')
# 运行回测系统
thestrats = cerebro.run()
# 获取回测结束后的总资金
portvalue = cerebro.broker.getvalue()
# 打印结果
print(f'结束资金: {round(portvalue, 2)}')

# cerebro的画图是呈现单只股票的买卖变化情况
# cerebro.plot(iplot=False) 


# 策略分析--绘制收益率变化图
# 读取benchmark数据
# path = '/Users/ryan/Desktop/quantchina/order-flow-alpha/data'
# file_name = 'market_return.csv'
# input_file = os.path.join(path, file_name)
# df_market = pd.read_csv(input_file, index_col=[0])

# df_market['datetime'] = df_market.index
# df_market['datetime'] = pd.to_datetime(df_market['datetime'])
# df_market.reset_index(drop=True, inplace=True)

# df_market.drop('money', axis=1, inplace=True)

# df_market['return_predict'] = df_market['close']/df_market['close'].shift(1)-1
# df_market['code'] = '399300'
# order = ['code', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'return_predict']

# df_market = df_market[order]

# thestrat = thestrats[0]
# pyfolio = thestrat.analyzers._pyfolio.get_analysis()
# returns = pyfolio['returns'].values()
# returns = pd.DataFrame(list(zip(pyfolio['returns'].keys(),pyfolio['returns'].values())), columns=['date','total_value'])


# sharpe = np.round(np.sqrt(252) * returns['total_value'].mean() / returns['total_value'].std(), 4)
# returns['total_value']=returns['total_value']+1
# returns['total_value'] = returns['total_value'].cumprod()
# annal_rtn = np.round(returns['total_value'].iloc[-1]**(252/len(returns))-1, 4)*100
# dd = 1-returns['total_value']/np.maximum.accumulate(returns['total_value'])
# end_idx = np.argmax(dd)
# start_idx = np.argmax(returns['total_value'].iloc[:end_idx])
# maxdd_days = end_idx-start_idx
# maxdd = np.round(max(dd), 4)*100

# print(f'Sharpe Ratio: {sharpe}')
# print(f'Annual Return: {annal_rtn}')
# print(f'Max drawdown: {maxdd}')

# df_market.rename(columns={'datetime':'date', 'return_predict':'market_return'}, inplace=True)
# returns = pd.merge(returns, df_market[['date','market_return']], on='date', how='left')

# returns['market_return'] = 1+returns['market_return']
# returns['market_return'] = returns['market_return'].cumprod()
# returns = returns.set_index('date')

# plt.plot(returns['market_return'], label='market_return', color='red', alpha=0.5)
# plt.plot(returns['total_value'], label='portfolio_return', color='blue', alpha=0.5)
# plt.title('100-Stocks Portfolio')
# plt.legend()
# plt.grid()
# plt.show()