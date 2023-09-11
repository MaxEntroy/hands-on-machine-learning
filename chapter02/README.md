## Chapter02 机器学习的数学基础

### 2.1 向量

- 向量的几何意义在于，可以看做空间的一个点，或者是原点指向该点的有向线段。
- 向量的内积(inner product or dot product)是标量。
- 向量的长度用范数(norm)表达。

### 2.2 矩阵

- 矩阵是数表，或者由数构成的矩阵元祖。
- 理解成向量组也没问题。
- 通过向量组，可以转化为线性方程组进行讨论。
- 矩阵的秩用来表示向量组当中线性无关向量的个数。

### 2.3 梯度(gradient)

- 动机
    - 机器学习模型求解，一般会转化为在一定约束条件下，某个函数在空间上的最值问题，这样的问题也叫做最优化问题。
    - 要求解这一优化问题，要分析函数值的变化趋势。
        - 如果我们知道了函数在某一点沿着任意方向，是上升还是下降
        - 就可以不断沿着函数值下降的方向走
        - 直到所有方向的函数值都上升为止

动机这里说的很明白，在某一个点，有无数个方向可以走。但是每个方向到底是使得函数值上升还是下降，或者上升多少，下降多少，都不一样。那么，这样就引出了梯度的概念，即梯度是这样一个方向，它使得函数值在该方向下降的最多。

所以，梯度也是描述函数变化速率和方向的工具。具体的数学定义如下：

$$ \nabla f(x) = \lgroup \frac{\partial f}{\partial x_{1}}, \frac{\partial f}{\partial x_{2}}\, ... , \frac{\partial f}{\partial x_{n}} \rgroup $$

很明显，这是一个vector，结合vector的物理含义，我们得到了一个空间中的点，当然此时我们考虑有向线段更容易理解。这个方向就是梯度的方向。

### 2.4. 凸函数(convex function)

梯度为 0 并不能说明函数达到极值点，比如$y = x^{3}$，该函数也不是凸函数。所以，凸函数一个好的性质是：凸函数可以找到极值点。

数学定义如下：考虑函数 $f{x}$，对任意的 $x_{1}, x_{2}, 0 < \alpha < 1$，有

$$ \alpha f(x_{1}) + (1 - \alpha)f(x_{2}) \geqslant f(\alpha x_{1} + (1 - \alpha) x_{2}) $$

则称 $f(x)$ 是凸函数。