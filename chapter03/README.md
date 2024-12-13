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

KNN details.
  - k and label num are prior knowledge. 
  - fit just stores the trainning data.
  - ```get_knn_labels``` is the core method. find k nearest neighbors.

最后再总结下
- 模型，其实是训练数据的抽象。有的可以转化为参数化模型，有的不行。
    - knn算是典型的非参数化模型，训练数据不转化。所谓的fit就是保存。
    - predict时，不通过公式计算。而是sample去trainning data里寻找。获取知识
- 思路有一个很巧妙的点，刚开始一直没反应过来。就是这个二维数字怎么计算相似度。
    - 它其实是把它放到欧式空间里面。欧式空间的核心是平面性。
    - 所以，对于每一个二维数字。我就把它拍平了。存成一维数组。这就是一个向量。
    - 然后计算向量中的相似度，来度量topK. 找到knn samples.
