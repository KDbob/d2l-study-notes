from mxnet import nd
from mxnet.gluon import nn


def comp_conv2d(conv2d, X):
    """
    # 定义一个函数来计算卷积层，它初始化卷积层权重，并对输入和输出做相应的升维和降维
    (主要是增删批量大小和通道数两个维度的信息)
    """
    conv2d.initialize()
    X = X.reshape((1, 1) + X.shape)  # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 排除不关心的前两维：批量和通道


def pading_test():
    """5.2.1填充"""
    X = nd.random.uniform(shape=(8, 8))
    conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
    print(comp_conv2d(conv2d, X).shape)  # expect:(8, 8)  8-3+1+2 = 8

    conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
    print(comp_conv2d(conv2d, X).shape)  # expect:(8, 8) 8-5+1+4 =8,8-3+1+2=8


def stride_test():
    """5.2.2步幅"""
    X = nd.random.uniform(shape=(8, 8))
    conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
    print(comp_conv2d(conv2d, X).shape)  # expect(4, 4)

    conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
    print(comp_conv2d(conv2d, X).shape)  # expect(4, 4)


if __name__ == '__main__':
    pading_test()
    stride_test()
