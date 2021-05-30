from mxnet import nd
from mxnet.gluon import nn


def pool2d(X, pool_size, mode='max'):
    """
    池化层的前向计算实现(与corr2d函数非常类似)
    """
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j:j + p_w].mean()
    return Y


def maxpool_avgpool_test():
    """二维最大池化层和平均池化层"""
    X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), 'avg'))


def padding_stride_test():
    """5.4.2. 填充和步幅"""
    X = nd.arange(16).reshape((1, 1, 4, 4))
    pool2d = nn.MaxPool2D(3)  # 默认步幅和池化窗口形状相同
    print(pool2d(X))
    pool2d = nn.MaxPool2D(3, padding=1, strides=2)  # 可以手动指定步幅和填充
    print(pool2d(X))
    pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))  # 也可以指定非正方形的池化窗口，并分别指定高和宽上的填充和步幅
    print(pool2d(X))


def channels_test():
    """5.4.3. 多通道"""
    X = nd.arange(16).reshape((1, 1, 4, 4))
    X = nd.concat(X, X + 1, dim=1)
    pool2d = nn.MaxPool2D(3, padding=1, strides=2)
    print(pool2d(X))    # 我们发现输出通道数仍然是2


if __name__ == '__main__':
    # maxpool_avgpool_test()
    # padding_stride_test()
    channels_test()