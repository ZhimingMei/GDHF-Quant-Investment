import pandas as pd
import numpy as np
import backtrader as bt

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
        if len(self.buy_lst)<2:    #! 需要修改该部分内容
            for secu in set(self.getdatanames()) - set(self.buy_lst):
                data = self.getdatabyname(secu)
                # 设置买入条件
                if (data.return_predict > 0):
                    # 买入价格为当前资金的多少倍
                    order_value = self.broker.getvalue()*0.1
                    order_amount = self.downcast(order_value/data.close[0], 100)
                    self.order = self.buy(data, size=order_amount, name=secu)
                    self.log(f'Buy:{secu}, price:{data.close[0]:.2f}, amount:{order_amount}')
                    self.buy_lst.append(secu)
                    print(self.buy_lst)
        
        elif self.position:
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
secu_lst = {'603160':{'start':'2017-02-14','end':'2022-03-01'},
            '000002':{'start':'2017-02-14','end':'2022-03-01'},
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

cerebro.plot(iplot=False)