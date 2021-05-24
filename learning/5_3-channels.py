import d2lzh as d2l
from mxnet import nd


def corr2d_multi_in(X, K):
    """
    实现含多个输入通道的互相关运算
    """
    # 首先沿着X和K的第0维（通道维）遍历。然后使用*将结果列表变成add_n函数的位置参数
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])


def corr2d_multi_in_out(X, K):
    """
    一个互相关运算函数来计算多个通道的输出
    """
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])


def corr2d_multi_in_out_1x1(X, K):
    """
    用全连接层中的矩阵乘法来实现 1×1卷积
    """
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)
    return Y.reshape(c_o, h, w)


def multi_in_test():
    X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
    print(corr2d_multi_in(X, K))


def multi_out_test():
    X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
    # 我们将核数组K同K+1（K中每个元素加一）和K+2连结在一起来构造一个输出通道数为3的卷积核
    K = nd.stack(K, K + 1, K + 2)
    print(K.shape)
    print(corr2d_multi_in_out(X, K))

def multi_1x1_test():
    X = nd.random.uniform(shape=(3, 3, 3))
    K = nd.random.uniform(shape=(2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    # 以上函数与之前实现的互相关运算函数corr2d_multi_in_out等价
    print((Y1 - Y2).norm().asscalar() < 1e-6)

if __name__ == '__main__':
    multi_in_test()
    multi_out_test()
    multi_1x1_test()