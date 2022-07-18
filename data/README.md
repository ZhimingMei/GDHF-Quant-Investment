# 数据储存

## 文件说明

示例：

`data.csv`: data 储存的是因子数据集，时间跨度为2022-01-01至2022-07-16

------------------------

**股票池：** 沪深300的成分股        
**时间周期：** 2017-01-01至2022-07-01             

[//]: # (在下方输入数据集的名称及文件描述)

**价值类因子**            
`value_factor.gz` value_factor 储存的是价值类有关因子          
字段：'S_VAL_PE'-市盈率, 'S_VAL_PB_NEW'-市净率, 'S_VAL_PS'-市销率, 'S_DQ_TURN'-换手率, 'S_DQ_MV'-流通市值, 'NET_ASSETS_TODAY'-当日净资产

**基础类因子**         
`basic_income.gz` basic_income 储存的是income sheet里面的数据       
字段：'OPER_REV'-营业收入, 'OPER_PROFIT'-营业利润, 'TOT_PROFIT'-总利润, 'LESS_SELLING_DIST_EXP'-销售费用, 'EBIT'-息税前利润, 'EBITDA'-息税折旧摊销前利润 

`basic_balance.gz` basic_balance 储存的是balance sheet里面的数据        
字段：'TOT_CUR_ASSETS'-总流通资本, 'TOT_CUR_LIAB'-总流通负债, 'TOT_ASSETS'-总资产, 'FIX_ASSETS'-固定资产          
 
`basic_ttm.gz` basic_ttm 储存的是TTM（最近12个月）数据              
字段：'NET_PROFIT_PARENT_COMP_TTM'-归属母公司净利润TTM, 'NET_CASH_FLOWS_OPER_ACT_TTM'-经营活动产生的现金流量净额TTM, 'OPER_REV_TTM'-营业收入TTM

***注意：balance sheet和income sheet的报告时间并不是日频的，需要做数据预处理***
