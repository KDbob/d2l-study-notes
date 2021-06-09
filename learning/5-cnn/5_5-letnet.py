import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

# 加载数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

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


def letnet_test():
    """5.5.1 letnet模型维度测试"""
    X = nd.random.uniform(shape=(1, 1, 28, 28))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)

def letnet_train():
    """5.5.2 获取数据和训练模型"""
    lr, num_epochs = 0.9, 5
    ctx = d2l.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


def letnet_ex():
    """练习：调整卷积窗口大小、输出通道数、激活函数、全连接层输出个数使准确率提升"""
    net = nn.Sequential()
    net.add(nn.Conv2D(channels=4, kernel_size=6, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=20, kernel_size=6, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            # Dense会默认将（批量大小，通道，高，宽）形状的输入转换成
            # （批量大小，通道 * 高 * 块）
            nn.Dense(120, activation='relu'),
            nn.Dense(84, activation='relu'),
            nn.Dense(10))

    lr, num_epochs = 0.3, 5
    ctx = d2l.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

if __name__ == '__main__':
    # letnet_train()
    letnet_ex()