# 7.14
net

//网络参考原型
//https://www.cnblogs.com/charlotte77/p/5629865.html
//https://blog.csdn.net/weixinhum/article/details/79326209
//前向计算简述为输入值*权重w+k1在用sigmoid得到隐含直，在重复一次上述过程得到输出值
//后向则输出层节点的梯度乘以自身值和（1-自身值），再乘以对应的隐含层节点的权重，根据学习率、输出层节点的梯度和隐含层节点的值计算权重更新量，并将其减去原来的权重最后更新权值，重新计算输出。
//数据来源http://yann.lecun.com/exdb/mnist/
