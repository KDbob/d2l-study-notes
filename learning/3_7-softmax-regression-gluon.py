import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 3.7.2 定义和初始化模型
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))  # 使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数

# 3.7.3 softmax和交叉熵损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

# 3.7.4 定义优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# 3.7.5 训练模型
if __name__ == '__main__':
    num_epochs = 5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)

