from mxnet import autograd, nd
from mxnet import gluon

from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet import init

# 3.3.1 生成数据集
num_exaples = 1000
num_inputs = 2
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_exaples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
# 3.3.2 读取数据集
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)  # 将训练数据的特征和标签组合
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)  # 随机读取小批量

# 3.3.3 定义模型
net = nn.Sequential()  # 定义一个模型变量net
net.add(nn.Dense(1))  # 在Gluon中，全连接层是一个Dense实例。我们定义该层输出个数为1

# 3.3.4 初始化模型参数
net.initialize(init.Normal(sigma=0.01))

# 3.3.5 定义损失函数
loss = gloss.L2Loss()  # 平方损失又称L2范数损失

# 3.3.6 定义优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# 3.3.7 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss:%f' % (epoch, l.mean().asnumpy()))

# 分别比较学到的模型参数和真实的模型参数
dense = net[0]
true_w, dense.weight.data()
true_b, dense.bias.data()