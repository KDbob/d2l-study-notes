from mxnet import autograd, nd

# 对函数  y=2x⊤x  求关于列向量  x  的梯度。我们先创建变量x，并赋初值
x = nd.arange(4).reshape((4, 1))

# 需要先调用attach_grad函数来申请存储梯度所需要的内存
x.attach_grad()

# 默认条件下MXNet不会记录用于求梯度的计算。我们需要调用record函数来要求MXNet记录与求梯度有关的计算
with autograd.record():
    y = 2 * nd.dot(x.T, x)


y.backward()
# ́对函数  y=2x⊤x  求关于列向量  x  的梯度为4x
print(x.grad)