# 数据储存

## 选择的因子（结果）
1. 价值类因子：S_VAL_PS, NET_ASSETS_TODAY
2. 基础类因子：OPER_REV_TTM, TOT_CUR_LIAB, TOT_ASSETS
3. 风险类因子：Skewness20, SharpeRatio20
4. 情绪类因子：VSTD20, WVAD
5. 财务与质量类因子：S_FA_ROE, S_FA_OPERATEINCOMETOEBT, S_FA_QUICK

## 文件说明

示例：

`data.csv`: data 储存的是因子数据集，时间跨度为2022-01-01至2022-07-16

------------------------

**股票池：** 沪深300的成分股        
**时间周期：** 2017-01-01至2022-07-01             

[//]: # (在下方输入数据集的名称及文件描述)

**股票行情数据**
`eod_price.gz` eod_price储存的是股票的日行情数据           
字段： 'S_DQ_PRECLOSE'-昨收盘价, 'S_DQ_OPEN'-开盘价, 'S_DQ_HIGH'-最高价, 'S_DQ_LOW'-最低价, 'S_DQ_CLOSE'-收盘价, 'S_DQ_VOLUME'-成交量（手）, 'S_DQ_AVGPRICE'-均价（VWAP，成交金额/成交量）

------------------------

**价值类因子**            
`value_factor.gz` value_factor 储存的是价值类有关因子          
字段：'S_VAL_PE'-市盈率, 'S_VAL_PB_NEW'-市净率, 'S_VAL_PS'-市销率, 'S_DQ_TURN'-换手率, 'S_DQ_MV'-流通市值, 'NET_ASSETS_TODAY'-当日净资产

**选择的因子：** S_VAL_PS, NET_ASSETS_TODAY

**基础类因子**         
`basic_income.gz` basic_income 储存的是income sheet里面的数据       
字段：'OPER_REV'-营业收入, 'OPER_PROFIT'-营业利润, 'TOT_PROFIT'-总利润, 'LESS_SELLING_DIST_EXP'-销售费用, 'EBIT'-息税前利润, 'EBITDA'-息税折旧摊销前利润 

`basic_balance.gz` basic_balance 储存的是balance sheet里面的数据        
字段：'TOT_CUR_ASSETS'-总流通资本, 'TOT_CUR_LIAB'-总流通负债, 'TOT_ASSETS'-总资产, 'FIX_ASSETS'-固定资产          
 
`basic_ttm.gz` basic_ttm 储存的是TTM（最近12个月）数据              
字段：'NET_PROFIT_PARENT_COMP_TTM'-归属母公司净利润TTM, 'NET_CASH_FLOWS_OPER_ACT_TTM'-经营活动产生的现金流量净额TTM, 'OPER_REV_TTM'-营业收入TTM

**选择的因子：** OPER_REV_TTM, TOT_CUR_LIAB, TOT_ASSETS

**风险类因子**           
`risk_factor.gz` risk_factor 储存的是风险类有关因子               
字段：'Variance20'-20日年化收益方差, 'Kurtosis20'-个股收益的20日峰度, 'Skewness20'-个股收益的20日偏度, 'SharpeRatio20'-20日夏普比率

**选择的因子：** Skewness20, SharpeRatio20

**情绪类因子**          
`trade_factor.gz` trade_factor 储存的是基于成交数据的情绪类因子           
字段：'VOL20'-20日平均换手率, 'VSTD20'-20日成交量标准差, 'TVMA20'-20日成交金额的移动平均值, 'WVAD'-威廉变异离散量

**选择的因子：** VSTD20, WVAD, 可以加上TVMA20（建议是删掉）

**财务与质量类因子**             
`fa_factor.gz` fa_factor 储存的是财务指标类因子以及质量因子             
字段：'S_FA_FCFF'-企业自由现金流量, 'S_FA_EPS_BASIC'-基本每股收益, 'S_FA_BPS'-每股净资产, 'S_FA_ORPS'-每股营业收入, 'S_FA_NETPROFITMARGIN'-销售净利率, 'S_FA_GCTOGR'-营业总成本/营业总收入, 'S_FA_ROE'-净资产收益率, 'S_FA_OPERATEINCOMETOEBT'-经营活动净收益/利润总额, 'S_FA_CATOASSETS'-流动资产/总资产, 'S_FA_CURRENT'-流动比率, 'S_FA_QUICK'-速动比率, 'S_FA_FATURN'-固定资产周转率, 'S_FA_OPTOLIQDEBT'-营业利润/流动负债, 'S_FA_PROFITTOOP'-利润总额/营业收入

**选择的因子：** S_FA_ROE, S_FA_OPERATEINCOMETOEBT, S_FA_QUICK

***注意：balance sheet和income sheet的报告时间并不是日频的，需要做数据预处理***
