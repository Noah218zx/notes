# 概念
逻辑回归（logistic regression）是一种分类（calssification）算法，逻辑回归的核心作用是**解决二分类问题（binary classification）**（后续可扩展到多分类），它通过对数据的建模，输出样本属于某一类别的“概率”，再根据概率阈值确定最终类别。例如：输出“用户购买商品的概率为85%”，若阈值设为50%，则判断该用户“会购买”。
在二分类问题中，y只能取0和1两个值，分别表示否和是
# 假设表示（Hypothesis Representation）
## 逻辑函数（sigmoid function）
这个函数可以将实数值映射到0-1的范围内，用来将预测值表示概率，公式如下：
$$
\begin{align*} & z = \boldsymbol{\theta}^\top \mathbf{x} \newline & g(z) = \dfrac{1}{1 + e^{-z}} \newline & h_{\boldsymbol{\theta}} (\mathbf{x}) = g ( \boldsymbol{\theta}^\top \mathbf{x} ) \end{align*}
$$
其中 $\boldsymbol{\theta} = \begin{bmatrix}\theta_0 \newline \theta_1 \newline \vdots \newline \theta_n\end{bmatrix}$ and $\mathbf{x} = \begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}$
$$
\begin{equation}  
h_{\boldsymbol{\theta}} (\mathbf{x}) = \dfrac{1}{1 + e^{-\boldsymbol{\theta}^\top \mathbf{x}}} = \dfrac{1}{1 + e^{-(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n)}}  
\end{equation}
$$
这是逻辑函数（sigmoid function）的图像
![[Pasted image 20251124142352.png]]
## 对数几率函数（logit function）
由sigmoid函数的概率输出可以推导出逻辑回归的另一种核心表达形式，这就需要引入**Logit函数**（也叫对数几率函数）
首先明确两个基础概念：

- **几率（Odds）**：正类概率与负类概率的比值，公式为$Odds = \frac{p(y=1|x)}{1-p(y=1|x)}$。例如，若p=0.8，则Odds=0.8/0.2=4，意味着“样本为正类的可能性是负类的4倍”；若p=0.5，则Odds=1，说明正、负类可能性相等。
    
- **Logit函数**：对几率取自然对数，即$Logit(p) = \log\left(\frac{p}{1-p}\right)$，
从而可以推导出$$\log\left(\frac{p(y=1|x;\theta)}{1-p(y=1|x;\theta)}\right) = \theta^\top x$$
这个等式**样本属于正类的对数几率，与特征的线性组合呈正比关系**。这也是逻辑回归被称为“广义线性模型”的原因——它通过Logit函数将概率与线性模型连接起来。![[Pasted image 20251124143951.png]]
# 决策边界（decision boundary）
有了概率p后，我们需要一个“**阈值（threshold）**”来判断样本类别，最常用的阈值是0.5：

- 若p ≥ 0.5，判断样本为正类（y=1）；
- 若p < 0.5，判断样本为负类（y=0）。 
我们可以假设函数的输出转换如下：
$$
\begin{align*}& h_{\boldsymbol{\theta}}(\mathbf{x}) \geq 0.5 \rightarrow y = 1 \newline& h_{\boldsymbol{\theta}}(\mathbf{x}) < 0.5 \rightarrow y = 0 \newline\end{align*}
$$
结合Sigmoid函数的特性，当p=0.5时，z=0，因此决策边界对应的线性方程为：

$$w_1x_1 + w_2x_2 + ... + w_nx_n + b = 0$$
即可以表示为：
$$
\begin{align*}& \boldsymbol{\theta}^\top \mathbf{x} \geq 0 \Rightarrow y = 1 \newline& \boldsymbol{\theta}^\top \mathbf{x} < 0 \Rightarrow y = 0 \newline\end{align*}
$$
这个阈值的**决策边界**是线$\boldsymbol{\theta}^\top \mathbf{x} = 0$，它分隔了y=0和y=1的区域。
例如：当特征只有2个（x1, x2）时，决策边界是一条直线；当特征有3个时，决策边界是一个平面；特征更多时，决策边界是高维超平面。

ROC曲线的AUC（曲线下面积）提供了所有可能分类阈值下的综合性能度量。当我们不知道最终将使用什么决策阈值时，AUC帮助我们在模型之间做出选择。[[评估指标（evaluation metrics）#AUC]]
# 成本函数（cost function）
和线性回归中的cost function一样，成本函数表示优化目标，逻辑回归的成本函数如下：
$$
\begin{align*}
J(\boldsymbol{\theta}) & = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}),y^{(i)}) \newline
& = \dfrac{1}{m} \sum_{i=1}^m \left[-y^{(i)} \log(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})) -(1-y^{(i)}) \log(1-h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}))\right]
\end{align*}
$$
代价可以写成：
$$
\begin{align*}  
& \mathrm{Cost}(h_{\boldsymbol{\theta}}(\mathbf{x}),y) = -\log(h_{\boldsymbol{\theta}}(\mathbf{x})) ; & \text{如果 y = 1} \newline  
& \mathrm{Cost}(h_{\boldsymbol{\theta}}(\mathbf{x}),y) = -\log(1-h_{\boldsymbol{\theta}}(\mathbf{x})) ; & \text{如果 y = 0}  
\end{align*}
$$即如果我们的正确答案'y'是0，那么当我们的假设函数也输出0时，代价函数将为0。如果我们的假设接近1，那么代价函数将趋向于无穷大。

如果我们的正确答案'y'是1，那么当我们的假设函数输出1时，代价函数将为0。如果我们的假设接近0，那么代价函数将趋向于无穷大。![[Pasted image 20251124145511.png]]
![[Pasted image 20251124145515.png]]
这个损失被称为**对数损失(log-loss)**。它也被称为**对数似然(log-likelihood)**或 **交叉熵(cross-entropy)** 损失。
# 梯度下降（gradient descent）
和线性回归一样，主要是用来得到最小的cost function：
$$
\begin{align*}& Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\boldsymbol{\theta}) \newline & \rbrace\end{align*}
$$
"共轭梯度(Conjugate gradient)"、"BFGS"和"L-BFGS"是更复杂、有时更快的优化θθ的方法，可以用来替代梯度下降

%% **正则化(Regularization)** 在逻辑回归中非常重要 同样极其重要的是，我们的模型需要具有泛化能力，以便在新数据上获得最佳预测，这正是我们创建模型的初衷。为了帮助实现这一点，重要的是不要过拟合我们的数据。因此，在目标函数中添加惩罚项，如用于稀疏性的L1L1​正则化和用于保持模型权重较小的L2L2​正则化，或者添加早停(early stopping)都可以在这方面提供帮助。%%
# 多类别分类（multiclass classification ）
## 一对多（one vs all）
核心思路：将多分类问题拆解为多个二分类问题。

- 假设共有K个类别，为每个类别训练一个逻辑回归模型（共K个模型）；
    
- 第i个模型将“类别i”视为正类，其余所有类别视为负类；
    
- 预测时，将样本输入所有K个模型，取概率最大的模型对应的类别作为最终结果。
    
优点：简单易实现，适合类别数量不多的场景；缺点：当类别不平衡时，模型性能可能受影响。
$$
\begin{align*}  
& y \in \lbrace1, 2 ... C\rbrace \newline  
& h_{\boldsymbol{\theta}}^{(1)}(\mathbf{x}) = p(y = 1 | \mathbf{x} ; \boldsymbol{\theta}^{(1)}) \newline  
& h_{\boldsymbol{\theta}}^{(2)}(\mathbf{x}) = p(y = 2 | \mathbf{x} ; \boldsymbol{\theta}^{(2)}) \newline  
& \cdots \newline& h_{\boldsymbol{\theta}}^{(C)}(\mathbf{x}) = p(y = C| \mathbf{x} ; \boldsymbol{\theta}^{(C)}) \newline  
& \mathrm{prediction} = \max_{c \in {1,2,...C}}( h_{\boldsymbol{\theta}} ^{(c)}(\mathbf{x}) )\newline  
\end{align*}
$$
## 多项逻辑回归（softmax regression）
### 概念
即将二元的y=0或1拓展到 y^(i)∈{1,…,C} 的情况，其中 C 是类别数量。
对于输入 x，我们希望假设函数能够估计 p(y=c|x) 的概率，其中 c={1,…,C}。因此，我们的假设函数会输出一个 C 维向量（元素和为1），包含所有 C 个类别的估计概率。具体来说，假设函数 h_Θ(x) 的形式如下（注意假设函数现在用粗体表示，因为它是一个向量）：
$$
\begin{equation}

\mathbf{h}_{\Theta}(\mathbf{x}) =

\begin{bmatrix}

h_{{\Theta}}^{(1)}(\mathbf{x}) \\

h_{{\Theta}}^{(2)}(\mathbf{x}) \\

\vdots \\

h_{{\Theta}}^{(C)}(\mathbf{x})

\end{bmatrix} =

\begin{bmatrix}

p(y = 1 | \mathbf{x}; {\Theta}) \\

p(y = 2 | \mathbf{x}; {\Theta}) \\

\vdots \\

p(y = C | \mathbf{x}; {\Theta})

\end{bmatrix} =

\frac{1}{ \sum_{c=1}^{C}{e^{\boldsymbol{\theta}^{(c)\top} \mathbf{x}} }}

\begin{bmatrix}

e^{\boldsymbol{\theta}^{(1)\top} \mathbf{x} } \\

e^{\boldsymbol{\theta}^{(2)\top} \mathbf{x} } \\

\vdots \\

e^{\boldsymbol{\theta}^{(C)\top} \mathbf{x} } \\

\end{bmatrix}

\end{equation}
$$
其中 θ^(1), θ^(2), ..., θ^(C) ∈ R^n 是模型的参数，分母项用于归一化，确保概率为一
### cost function
Softmax回归的代价函数如下，其中 1{.} 是指示函数（当条件为真时为1，为假时为0）：
$$
\begin{align}

J(\Theta) = - \left[ \sum_{i=1}^{m} \sum_{c=1}^{C}  \mathbb{1}_{\left\{y^{(i)} = c\right\}} \log p(y^{(i)} = c | \mathbf{x}^{(i)} ; \Theta)\right]

\end{align}
$$
