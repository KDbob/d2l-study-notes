# 3.2 线性回归的从零开始实现

from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

# 3.2.1 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)      # 模拟真实值，利用正态分布随机函数


# 3.2.2 读取数据集
# 本函数已保存在d2lzh包中方便以后使用
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)  # take函数根据索引返回对应元素


batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break


# 3.2.3 初始化模型参数
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
w.attach_grad()
b.attach_grad()


# 3.2.4 定义模型、损失函数、优化算法
# 定义模型
def linreg(X, w, b):
    return nd.dot(X, w) + b


def squared_loss(y_hat, y):
    '''
    定义损失函数(平方损失)
    :param y_hat: 预测函数
    :param y: 真实函数
    :return: 返回值的形状也和y_hat相同★
    '''
    # 需要把真实值y变形成预测值y_hat的形状。以下函数返回的结果也将和y_hat的形状相同
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    '''
    定义优化算法
    :param params: 向量中的每个参数
    :param lr: 学习率
    :param batch_size: 批量大小
    :return:
    '''
    for param in params:
        param[:] = param - lr * param.grad / batch_size


lr = 0.3
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）
    # X和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([b, w], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
        # print([b, w])
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))
