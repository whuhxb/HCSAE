There are two methods to train the SAE features:
1. Only using the training sample patches to train the SAE features, refer to the codes in UCM coder folder if necessary.
2. Using the whole sample patches to train the SAE features, this experiment mainly adopt this strategy.


The code to run the HCSAE on Google dataset:
First run GoogleSingleLayer.m, then run GoogleSecondLayer.m. 


训练SAE提取特征有两种方式
1. 仅用训练样本切块训练SAE特征， 执行以下第一个代码
2. 用全部影像切块训练SAE特征，执行以下第二个代码，一般用这个

对于Google数据集，第一层CSAE，运行GoogleSingleLayer.m至最优，然后第二层CSAE运行GoogleSecondLayer.m。