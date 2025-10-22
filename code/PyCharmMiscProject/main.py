%matplotlib inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# a = torch.zeros(2,3)
# print(a)
#
# b = torch.ones(2,3)
# print(b)
#
# c = torch.randn(2,3)
# print(c)
#
# numpy_array = np.array([[1,2],[3,4]])
# tensor = torch.from_numpy(numpy_array)
# print(tensor)
#
# device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
# d = torch.randn(2,3,device=device)
# print(d)

# # 创建一个需要梯度的张量
# tensor_requires_grad = torch.tensor([1.0], requires_grad=True)
#
# # 进行一些操作
# tensor_result = tensor_requires_grad * 2
#
# # 计算梯度
# tensor_result.backward()
# print(tensor_requires_grad.grad)  # 输出梯度

# y = a.view(5)
# z = a.view(-1,5)
# print(a.size(),y.size(),z.size())

# x = torch.arange(1,3).view(1,2)
# print(x)
# y = torch.arange(1,4).view(3,1)
# print(y)
# print(x+y)

# a = torch.ones(5)
# b = a.numpy()
# print(a,b)
#
# a += 1
# print(a,b)
#
# b += 1
# print(a,b)

# x = torch.ones(2,2,requires_grad=True)
# print(x)
# print(x.grad_fn)
#
# y = x + 2
# print(y)
# print(y.grad_fn)
#
# z = y * y * 3
# out = z.mean()
# print(z,out)
#
# out.backward()
# print(x.grad)

# a = torch.ones(1000)
# b = torch.ones(1000)
#
# start = time()
# # c = torch.zeros(1000)
# # for i in range(1000):
# #     c[i] = a[i] + b[i]
# d = a + b
# print(time()-start)

num_inputs = 2
num_examples = 1000
true_w = [2,-3,4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,dtype=torch.float32)
labels = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()), dtype=torch.float32)

print(features[0],labels[0])

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
