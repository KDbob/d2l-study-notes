import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

# 通过Sequential类来实现LeNet模型
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # Dense会默认将（批量大小，通道，高，宽）形状的输入转换成
        # （批量大小，通道 * 高 * 块）
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))

if __name__ == '__main__':
    X = nd.random.uniform(shape=(1, 1, 28, 28))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)
