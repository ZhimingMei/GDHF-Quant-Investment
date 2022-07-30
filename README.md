# GDHF-Quant-Investment
Codes and documents of the competition held by HOUFANG Investment

## 题目

### 基于机器学习的多因子A股市场量化选股策略研究

要求:

- 提供完整的选股逻辑和分析过程;         
- 采用机器学习的方法，进行因子优选，构造多因子选股策略; 
- 策略分析包括样本内和样本外，覆盖时间足够长，建议从 2005 年开始分析;
- 股票数量需要 50 只以上，同时可以分析每期持股数量 20,30,50,100 的情况;

## 项目进度

需要更改项目进度可以点进编辑，然后在`- [] `里面写x，即`- [x] `，就会变成已完成         
示例：                  
`- [x] 泳池派队`
- [x] 泳池派队

### 任务列表
- [x] **数据预处理**
  - [x] 准备因子数据集
  - [x] 根据因子表现完成因子筛选
  - [x] 特征工程进行因子的进一步筛选
  
- [x] **模型构建**
  - [x] 机器学习模型构建
 
- [x] **模型优化** 

## 项目结构
```
    项目结构如下
    ├─data 从数据库中获取的数据文件
    │  ├─data_cleaned 清洗后的数据文件
    │  │  └─README.md 见描述文件
    │  ├─eod_price.gz 股票日行情数据
    │  ├─basic_xx.gz 基础类因子数据
    │  ├─value_factor.gz 价值类因子数据   
    ├─code 项目代码
    │  ├─factor
    │  │  ├─get_data.ipynb 基于wind和joinquant数据库的数据提取
    │  │  ├─factor_analysis.ipynb 因子分析
    │  │  └─
    │  ├─机器学习
    │  └─项目优化
    └─README.md 项目整体描述文档
```

## 参考资料

[Markdown语法介绍](https://www.runoob.com/markdown/md-tutorial.html)      

[git操作介绍](http://www.ruanyifeng.com/blog/2018/10/git-internals.html)      

[人工智能选股之循环神经网络----华泰人工智能系列之九](https://mp.weixin.qq.com/s/YGFZRqxerpplXzv2FqNGqQ)
