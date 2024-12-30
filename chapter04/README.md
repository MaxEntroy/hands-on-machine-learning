# chapter04 Linear regression.

## Basic

- total sample = feature + tag
- feature: Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population
- tag: Price.

我们的任务是学习到tag和features的关系。

- 目前来看，推荐系统的sample，行为特征类似click, view。这个到底算什么？
- 因为看score算是最终的tag。类比这里的price 但问题是，那click, view这些又算什么？

刚想明白了：
- total sample = feature + tag
- feature: user features + item features + context features.
- tag: ctr
- 这里的问题是，total sample = feature + tag，这个tag应该是ctr。那这个东西从哪拿呢？
- ctr = click / view. click/view是用户的行为，所以对于total sample来说ctr是tag，没问题。
- 但是它以另一种形式体现出来。所以，推荐系统里面也说tag是click/view这些东西。

## Feature

这里稍微延伸一下，feature又是什么？RS 里面的这个raw feature/result feature又是什么？为什么要这么做。

>In machine learning and pattern recognition, a feature is an individual measurable property or characteristic of a data set.[1] 
Choosing informative, discriminating, and independent features is crucial to produce effective algorithms for pattern recognition, classification, and regression tasks.

From the definition of feature from wikipedia, we know that feature is an individual mesurable property of a data set which is informative, discriminating, crucial for model.

这个定义解释了feature，简单说，feature这个概念必须相对特定的模型来说。对特定模型来说，有用的数据特性，可以认为是特征。


## Feature Engineering

这里回答另一个问题，就是raw feature/result feature，这些东西又是什么？为什么要这么做。

wikipedia把feature extraction redirect to feature engineering，证明这两说的是一回事。

>Feature engineering is a preprocessing step in supervised machine learning and statistical modeling[1] which transforms raw data into a more effective set of inputs.
By providing models with relevant information, feature engineering significantly enhances their predictive accuracy and decision-making capability.

第一句话说的是feature extraction. transform raw feature to a more effective features. 这里有些我理解是必须的，比如non-numeric types，还有处理不同的量纲。

第二句话说的是feature selection. By providing models with relevant information, 这个relevant information很重要。血糖预测，身高，体重，血型，国籍，这么多特征，到底哪个对预测有作用呢？
这个是feature selection做的事情。

## Model training.

- 机器学习派别 vs 统计派别
  - 核心差异在于：参数的求解思路。注意，这里区别的是思路。不是方法。不管机器学习派别还是统计派别，最后参数求解的方法都是转化为最优化求解。
  - 机器学习派别求解思路：确定线性模型。但是对于参数求解没有先验知识。
    - 定义loss function. 这个是参数的函数。(X, Y)在这个函数中被是做常数。
    - 通过求解loss function的最小值，拿到参数的解。
  - 统计派别求解思路：确定线性模型。但是对于参数求解有先验支持。
    - **先验信息**，噪声服从某一个分布。此时，可以拿到y的分布。
    - 注意，此时有(x, y)观测值。那就说明这个事情发生了，likelihood function代入这组观测值，应该可以拿到最大值。
    - 似然函数是参数的参数。(X, Y)在这个函数中被做看做常数。
    - 通过求解likelihood function的最大值，拿到参数的解。
  - 思路的区别，其实就是这个optimize function的区别，前者loss function. 后者likelihood function.

y = a1x1 + a2x2 + a3x3 + b，theta = (a1, a2, a3), X = (x1, x2, x3)。这么写的问题是，形式不统一。

上面的式子，可以转化为：

y = a1x1 + a2x2 + a3x3 + a4*1, theta = (a1, a2, a3, b), X = (x1, x2, x3, 1)

这样便于求解全部参数。否则 b = y - theta*x，需要两步。这也是代码里对于x扩充的原因

## Frequentist vs Bayesian

这个到不是重点，不过我也展开下。还是对于linear regression这个模型来说。

参数求解的思路是一样的。都是likelihood function. 

- frequentist
  - 对于参数没有先验，噪声有。借此，可以拿到一个y的分布。可以表达成likelihood function.
- bayesian
  - 对于参数有先验。还是可有你妈到一个y的分布。此时是P(y|w), w是有先验的。也可以表达成likelihood function.
- 这二者的likelihood function不一样。