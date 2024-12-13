## Chapter03 KNN算法

###  原理

### 实践

- 对于samples数据的处理，本质二维的0-1 数表，实际存储的时候，拍成了一维的结构，这点注意。
- 具体的数据集合，1000 个样本
    - 每一个 X，原本是28 * 28的矩阵，拍成一维矩阵后是 784 个元素
    - 每一个 Y 就是一个具体的图像值

```python
np.random.seed(0)
shuffle_indices = np.random.permutation(np.arange(len(samples)))
samples = samples[shuffle_indices]
labels = labels[shuffle_indices]
```
shuffle的逻辑，python写法很简单。最后的赋值类似a[0] = a[9]

KNN的实现细节
  - k and label num are prior knowledge. 