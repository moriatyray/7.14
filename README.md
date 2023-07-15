# 7.14
net

//网络参考原型
//https://www.cnblogs.com/charlotte77/p/5629865.html
//[https://blog.csdn.net/weixinhum/article/details/79326209](https://blog.csdn.net/weixinhum/article/details/74908042?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168925392716800188510591%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=168925392716800188510591&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-3-74908042-null-null.268^v1^koosearch&utm_term=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0&spm=1018.2226.3001.4450)https://blog.csdn.net/weixinhum/article/details/74908042?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168925392716800188510591%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=168925392716800188510591&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-3-74908042-null-null.268^v1^koosearch&utm_term=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0&spm=1018.2226.3001.4450
//前向计算简述为输入值*权重w+k1在用sigmoid得到隐含直，在重复一次上述过程得到输出值
//后向则输出层节点的梯度乘以自身值和（1-自身值），再乘以对应的隐含层节点的权重，根据学习率、输出层节点的梯度和隐含层节点的值计算权重更新量，并将其减去原来的权重最后更新权值，重新计算输出。
//数据来源http://yann.lecun.com/exdb/mnist/
//7.14反向传播好像算的有问题，等一下
//
