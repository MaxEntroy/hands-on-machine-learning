import matplotlib.pyplot as plt
import numpy as np
import os

# 读入mnist数据集
m_x = np.loadtxt('mnist_x.txt', delimiter=' ')
m_y = np.loadtxt('mnist_y.txt')

print (m_x)

# 数据集可视化
data = np.reshape(np.array(m_x[0], dtype=int), [28, 28])
plt.figure()

plt.imshow(data, cmap='gray')
plt.show()

print (data)