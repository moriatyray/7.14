// 网络搭建
// 数据读取(输入)
// 训练部分
// 输出

//网络参考原型
//https://www.cnblogs.com/charlotte77/p/5629865.html
//https://blog.csdn.net/weixinhum/article/details/74908042?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168925392716800188510591%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=168925392716800188510591&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-3-74908042-null-null.268^v1^koosearch&utm_term=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0&spm=1018.2226.3001.4450
//前向计算简述为输入值*权重w+k1在用sigmoid得到隐含直，在重复一次上述过程得到输出值
//后向则输出层节点的梯度乘以自身值和（1-自身值），再乘以对应的隐含层节点的权重，根据学习率、输出层节点的梯度和隐含层节点的值计算权重更新量，并将其减去原来的权重最后更新权值，重新计算输出。
//数据来源http://yann.lecun.com/exdb/mnist/

#include "time.h"
#include <iostream>
#include"wayland-util.h"
#include <stdio.h>
//#include"rubbish.h"
//#include"head.h"
//#include"net.h"
using namespace std;
const int IPNNUM = 784;
const int HDNNUM = 100;
const int OPNNUM = 10;

/// ///////////////////////网络搭建/////////////////////////////////////////////
//权重
class node
{
public:
	double value; //存储结点的状态
	double *W = NULL;    //结点到下一层的权值

	void initNode(int num);//初始化函数，必须调用以初始化权值个数
	~node();     //析构函数，释放掉权值占用内存
};

void node::initNode(int num)
{
	W = new double[num];//分配空间
	srand((unsigned)time(NULL));//随机数种子的初始化
	for (int i = 0; i < num; i++)//给权值赋一个随机值
	{
		W[i] = rand() % 100 / double(1000);//100内随机数除以1000得0-0.099
		if (rand() % 2)
		{
			W[i] = -W[i];
		}
	}
}

node::~node()
{
	delete[] W;
}

//网络类，描述神经网络的结构并实现前向传播以及后向传播
class net
{
public:
	node inlayer[IPNNUM]; //输入层//成员变量
	node hidlayer[HDNNUM];//隐含层
	node outlayer[OPNNUM];//输出层

	double yita = 0.1;//学习率η
	double d1;//输入层偏置项
	double d2;//隐含层偏置项
	double Tg[OPNNUM];//训练目标
	double O[OPNNUM];//网络实际输出

	net();//构造函数，用于初始化各层和偏置项权重
	double sigmoid(double z);//激活函数
	double Loss();//损失函数，输入为目标值
	void forwardpropagation(double *input);//前向传播,输入为输入层节点的值
	void backpropagation(double *T);//反向传播，输入为目标输出值

};

net::net()
{
	//输入层和隐含层偏置项权值，给一个随机值
	srand((unsigned)time(NULL));
	d1 = rand() % 100 / double(100);
	d2 = rand() % 100 / double(100);
	//初始化输入层到隐含层节点权重
	for (int i = 0; i < IPNNUM; i++)
	{
		inlayer[i].initNode(HDNNUM);
	}
	//初始化隐含层到输出层节点权重
	for (int i = 0; i < HDNNUM; i++)
	{
		hidlayer[i].initNode(OPNNUM);
	}
}
//激活函数 引入非线性因素
double net::sigmoid(double z)
{
	return 1 / (1 + exp(-z));
}
//损失函数
double net::Loss()
{
	double loss = 0;
	for (int i = 0; i < OPNNUM; i++)
	{
		loss += pow(O[i] - Tg[i], 2);
	}
	return loss / OPNNUM;
}
//前向传播
void net::forwardpropagation(double *input)
{
	for (int iNNum = 0; iNNum < IPNNUM; iNNum++)//输入层节点
	{
		inlayer[iNNum].value = input[iNNum];
	}
	for (int hNNum = 0; hNNum < HDNNUM; hNNum++)//算出隐含层结点的值
	{
		double z = 0;
		for (int iNNum = 0; iNNum < IPNNUM; iNNum++)
		{
			z += inlayer[iNNum].value*inlayer[iNNum].W[hNNum];
		}
		z += d1;//
		hidlayer[hNNum].value = sigmoid(z);
	}
	for (int oNNum = 0; oNNum < OPNNUM; oNNum++)//算出输出层结点的值
	{
		double z = 0;
		for (int hNNum = 0; hNNum < HDNNUM; hNNum++)
		{
			z += hidlayer[hNNum].value*hidlayer[hNNum].W[oNNum];
		}
		z += d2;//加上偏置项
		O[oNNum] = outlayer[oNNum].value = sigmoid(z);
	}
}
//反向传播，

void net::backpropagation(double *T)
{
	for (int i = 0; i < OPNNUM; i++)
	{
		Tg[i] = T[i];
	}
	for (int iNNum = 0; iNNum < IPNNUM; iNNum++)//更新输入层权重
	{
		for (int hNNum = 0; hNNum < HDNNUM; hNNum++)
		{
			double y = hidlayer[hNNum].value;
			double loss = 0;
			for (int oNNum = 0; oNNum < OPNNUM; oNNum++)
			{
				loss += (O[oNNum] - Tg[oNNum])*O[oNNum] * (1 - O[oNNum])*hidlayer[hNNum].W[oNNum];
			}
			inlayer[iNNum].W[hNNum] -= yita * loss*y*(1 - y)*inlayer[iNNum].value;
		}
	}
	for (int hNNum = 0; hNNum < HDNNUM; hNNum++)//更新隐含层权重
	{
		for (int oNNum = 0; oNNum < OPNNUM; oNNum++)
		{
			hidlayer[hNNum].W[oNNum] -= yita * (O[oNNum] - Tg[oNNum])*
				O[oNNum] * (1 - O[oNNum])*hidlayer[hNNum].value;
		}
	}
}



/////////////////////////////////////////////图像获取/////////////////////////////////

class ImgData//单张图像
{
public:
	unsigned char tag;
	double data[IPNNUM];
	double label[OPNNUM];
};

class getImg
{
public:
	ImgData* mImgData;
	void imgTrainDataRead(const char *datapath, const char *labelpath);
	~getImg();
};

void getImg::imgTrainDataRead(const char *datapath, const char *labelpath)
{

unsigned char readbuf[4];//信息数据读取空间
FILE *f = fopen(datapath, "rb");

fread(readbuf, 1, 4, f);//读取文件标志位
fread(readbuf, 1, 4, f);//读取数据集图像个数
int sumOfImg = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像个数
fread(readbuf, 1, 4, f);//读取数据集图像行数
int imgheight = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像行数
fread(readbuf, 1, 4, f);//读取数据集图像列数
int imgwidth = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像列数
mImgData = new ImgData[sumOfImg];
unsigned char *data = new unsigned char[IPNNUM];
for (int i = 0; i < sumOfImg; i++)
{
    fread(data, 1, IPNNUM, f);//读取数据集图像列数
    for (int px = 0; px < IPNNUM; px++)//图像数据归一化
    {
        mImgData[i].data[px] = data[px] / (double)255 * 0.99 + 0.01;
    }
}
delete[] data;
fclose(f);
	f = fopen(labelpath, "rb");

fread(readbuf, 1, 4, f);//读取文件标志位
fread(readbuf, 1, 4, f);//读取数据集图像个数
sumOfImg = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像个数
for (int i = 0; i < sumOfImg; i++)
{
    fread(&mImgData[i].tag, 1, 1, f);//读取数据集图像列数
    for (int j = 0; j < 10; j++)
    {
        mImgData[i].label[j] = 0.01;
    }
    mImgData[i].label[mImgData[i].tag] = 0.99;
}
fclose(f);

}

getImg::~getImg()
{
	delete[]mImgData;
}
/////////////////////////////////训练

    // time 表示当前轮数；
    // mnet 是指向神经网络对象的指针；
    // mImg 是指向输入图像数据对象的指针。

void Accuracyrate(int time, net *mnet, getImg *mImg)//精确率评估
{
	double tagright = 0;//正确个数统计
	for (int count = 0; count < 10000; count++)//遍历 10000 个样本进行评估
	{
		mnet->forwardpropagation(mImg->mImgData[count].data);//前向传播
		double value = -100;
		int gettag = -100;
		for (int i = 0; i < 10; i++)
		{
			if (mnet->outlayer[i].value > value)//概率最大的是最有可能的预测结果
			{
				value = mnet->outlayer[i].value;
				gettag = i;
			}
		}
		if (mImg->mImgData[count].tag == gettag)
		{
			tagright++;
		}
	}
	//mnet.printresual(0);//信息打印
	cout << "第" << time + 1 << "轮:  ";
	cout << "正确率为:" << tagright / 10000 << endl;
}

int main()
{
	getImg mGetTrainImg;
	mGetTrainImg.imgTrainDataRead("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	getImg mGetTestImg;
	mGetTestImg.imgTrainDataRead("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	net mnet;//神经网络对象
	for (int j = 0; j < 10; j++)
	{
		for (int i = 0; i < 6000; i++)
		{
			mnet.forwardpropagation(mGetTrainImg.mImgData[i].data);//前向传播
			mnet.backpropagation(mGetTrainImg.mImgData[i].label);//反向传播
		}
		Accuracyrate(j,&mnet, &mGetTestImg);
	}
	cout << "finish\n"; 
}
