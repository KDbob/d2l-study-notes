from d2lzh import corr2d
from mxnet.gluon import nn
from mxnet import nd, autograd


class Conv2D(nn.Block):
    def __init__(self, channels, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(channels, 1) + kernel_size)
        self.bias = self.params.get('bias', shape=(channels,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


def learning_a_kernal():
    # 图像中物体边缘检测
    X = nd.ones((6, 8))
    X[:, 2:6] = 0
    K = nd.array([[1, -1]])
    Y = corr2d(X, K)

    # 5.1.4. 通过数据学习核数组
    # 构造一个输出通道数为1（将在“多输入通道和多输出通道”一节介绍通道），核数组形状是(1, 2)的二
    # 维卷积层
    conv2d = nn.Conv2D(1, kernel_size=(1, 2))
    conv2d.initialize()

    # 二维卷积层使用4维输入输出，格式为(样本, 通道, 高, 宽)，这里批量大小（批量中的样本数）和通
    # 道数均为1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))

    for i in range(10):
        with autograd.record():
            Y_hat = conv2d(X)
            l = (Y_hat - Y) ** 2
        l.backward()
        conv2d.weight.data()[:] -= conv2d.weight.grad() * 3e-2
        if (i + 1) % 2 == 0:
            print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))


class Conv2D_ex3(nn.Block):
  """
    试着对我们自己构造的Conv2D类进行自动求梯度，会有什么样的错误信息？在该类的forward函数里，将corr2d函数替换成nd.Convolution类使得自动求梯度变得可行。
    - **data**: *(batch_size, channel, height, width)*
    - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_height, out_width)*.
  """

  def __init__(self, channels, kernel_size, **kwargs):
    super(Conv2D_ex3, self).__init__(**kwargs)
    self.weight = self.params.get(
        'weight', shape=(
            channels,
            1,
        ) + kernel_size)
    self.bias = self.params.get('bias', shape=(channels, ))
    self.num_filter = channels
    self.kernel_size = kernel_size

  def forward(self, x):
    return nd.Convolution(
        data=x, weight=self.weight.data(), bias=self.bias.data(), num_filter=self.num_filter, kernel=self.kernel_size)


def learning_a_kernal_ex3():
    print('\ne.x.3:')
    X = nd.ones((6, 8))
    X[:, 2:6] = 0
    K = nd.array([[1, -1]])
    Y = corr2d(X, K)

    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    conv2d = Conv2D_ex3(1, kernel_size=(1, 2))
    conv2d.initialize()
    for i in range(10):
      with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y)**2
        if i % 2 == 1:
          print('batch %d, loss %.3f' % (i, l.sum().asscalar()))
      l.backward()
      conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()


if __name__ == '__main__':
    learning_a_kernal_ex3()
