## Chapter01 初探机器学习

### 1.1 两只手和四条腿

- 两只手：两大类任务
    - 预测
    - 决策
- 四条腿：四大类技术
    - 搜索
    - 推理
    - 学习
    - 博弈

### 1.2 机器学习是什么

自然语言的表达如下：

Herbert Simon: A program is said to learn from experience E with respect to some class of task T and performance measure P. If its performance at tasks in T, as measured by P, **improves** with experience E, then it is machine learning.

简言之，机器学习是系统通过经验提升任务性能的过程。一组学习任务可以由一个三元组表示<T,P,E>

数学语言的表达如下：

对应为一个优化问题，即针对某一类预测任务T，其数据集为D，对于一个机器学习模型f，预测任务的性能指标可以通过一个函数(loss function)来表示，那么机器学习的过程则是在一个给定的模型空间F中，寻找可以最大化性能指标的预测模型f。

我觉得数学语言的表达更精确一点，机器学习，本质是寻找一个模型，或者确定优化参数的问题。我们看它的形式

$$f^{*} = arg max_{f \in F} T(D, F)$$

其和<T,P,E>的对应来自于，数据给出，任务给出，我们通过寻找评价指标的最优值，来确定的我们的模型参数，这个是核心的理解。即通过寻找最大化性能指标的预测模型 $f^{*}$ 来完成机器学习的任务。

为什么说它是non-explicit programming？ 因为我们通常事先不知道 $f^{*}$ ，所以无法直接实现它。比如，分类，聚类这些任务都不能事先知道最好的分类和聚类。

既然是non-explicit programming，不是直接解决问题的算法，那么机器学习算法又代表着什么呢？

$$ f_{0} \rightarrow f_{1} \rightarrow f_{2} \rightarrow ... \rightarrow f^{*} $$

上述寻找最优模型的过程就是机器学习，机器学习算法对应着从 $f_{i}$ 迭代到 $f_{i + 1}$ 的程序。

### 1.3. 机器学习分类

两种分类方式，分类和是否参数化。

- 任务分类
    - Supervised learning: 模型的任务是根据数据特征来预测label. feature + label = sample
    - Unsupervised learning: no label
    - reinforment learning: 关心的是决策问题，不是预测问题。
- 模型是否参数化
    - Parametric model: 在一套具体的模型族(model family)内，每一个具体的模型，可以用一个具体的参数向量来唯一确定。因此确定了参数向量，也就确定了模型。
    - Non-parametric model: 模型并不是由一组参数向量决定，其训练算法也不是更新模型的参数，而是由具体的规则，直接在模型空间中寻找模型实例。

###  1.4. 机器学习奏效的本质

generation ability被用来描述，一个智能模型在没见过的数据上的预测性能。一般用泛化误差表示。为什么机器学习在有限数据的训练后，就可以在没见过的数据上做出一定精度的预测？机器学习的底层是数理统计，其基本原理是，相似的数据拥有相似的标签。所以，如果发现预测数据的和已知数据相似，那么预测数据的标签可能和已知数据一致。

### 1.5. 机器学习模型的天赋(价值观)

inductive bias指的是模型对问题的先验假设。也就是所谓的先验知识，价值观，类似你带着某一类价值观看问题。深刻的价值观，能更好的看到问题的本质。机器学习也一样。

### 1.6. 机器学习的限制

- 数据限制
- 泛化能力限制
- 使用形态限制