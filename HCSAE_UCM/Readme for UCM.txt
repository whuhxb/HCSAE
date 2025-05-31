There two methods to train the SAE features:

1. Only using the training sample patches to train the SAE features, and execute the following 1 code
2. Using the whole sample patches to train the SAE features, and execute the following 2 code

There are two codes to run the HCSAE on UCM dataset:
1. Only using the training sample patches to train SAE, First run UCMSingleLayer_first.m to adjust the parameter, which will automatically run the UCMSingleLayer_second.m. 
Then, run UCMSecondLayer.m to the optimal. UCMSingleLayer_second.m and UCMSecondLayer.m adopt the fixed random 
number to train the softmax classifer.

2. Using the whole image sample patches to train SAE, first run UCMSingleLayer, then run UCMSecondLayer.m. 

训练SAE提取特征有两种方式
1. 仅用训练样本切块训练SAE特征， 执行以下第一个代码
2. 用全部影像切块训练SAE特征，执行以下第二个代码，一般实验用这个方案

两套执行代码：
1. 为了方便第一层CSAE调参，遍历调节第一层CSAE的hidden size，运行UCMSingleLayer_first.m, 会自动调用UCMSingleLayer_second.m。
然后，再运行第二层CSAE的UCMSecondLayer.m调参至最优。UCMSingleLayer_second.m和UCMSecondLayer.m采用了固定的随机数用于
两层网络的分类。但其实，是否固定随机数跟SAE的Unsupervised feature learning没有关系，因为SAE特征学习阶段是没有任何label的。

2. 手动调参：对于UCM数据，第一层CSAE，运行UCMSingleLayer.m至最优，第二层CSAE运行UCMSecondLayer.M。这里是试验阶段，
UCMSingleLayer.m没有采用固定的随机数用于分类器。