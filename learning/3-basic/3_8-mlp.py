import d2lzh as d2l
from mxnet import autograd, nd


def xyplot(x_vals, y_vals, name):
    """定义一个绘图函数"""
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')

if __name__ == '__main__':
    x = nd.arange(-8.0, 8.0, 0.1)
    x.attach_grad()
    with autograd.record():
        y = x.relu()
    xyplot(x, y, 'relu')
    d2l.plt.show()