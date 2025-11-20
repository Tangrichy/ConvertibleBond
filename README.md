# 可转债高频交易
> 基于__Kronos__的应用

关于具体Kronos的使用查看 [github](https://github.com/shiyu-coder/Kronos);

模型使用的微调代码：[Finetune](https://github.com/shiyu-coder/Kronos/tree/master/finetune_csv);

微调是基于模型 __Kronos-Tokenizer-base__ 和分词器 __NeoQuasar/Kronos-small__，根据Issues建议使用小型模型表现更好；

## 环境配置
```{bash}
pip install -r requirements.txt
```
如果加载 `from model import Kronos, KronosTokenizer, KronosPredictor` 出现问题，尝试删除`model`文件中的两个`init`文件

## 文件及使用说明

- model: 包含全部 __Kronos__ 神经网络代码
- prediction_all: 每次输入一支可转债，预测整个可转债的全部时间价格
- prediction_each: 每次输入一支可转的10行，预测下一个时间的价格，按照10行滚动


已经完成五支债券的微调模型，分别用五个已经完成微调的模型对债券进行预测，五个预测值求平均值，作为预测价格，如果预测值大于当前的收盘价，即标注为买入，如果小于当前的收盘价，即不交易

