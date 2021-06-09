import sys
import time

from d2lzh import utils as d2l
from mxnet.gluon import data as gdata

# 3.5.1 获取数据集
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
# 第一次会下载文件，下载的数据集位置：'/Users/bob/.mxnet/datasets/fashion-mnist'
# print(len(mnist_train), len(min))
# 获取第几个数据集
# X, y = mnist_train[0:9]

# d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))

# 3.5.2. 读取小批量
batch_size = 256    # 测试128时，性能提升
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4     # 测试2时，性能下降
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size=batch_size, shuffle=True, num_workers=num_workers)
gdata.DataLoader(mnist_test.transform_first(transformer), batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 测试读取性能
start = time.time()
for X, y in train_iter:
    continue
'%.2f sec' % (time.time() - start)
print('end')
