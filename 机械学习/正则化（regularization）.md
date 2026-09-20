## L2正则化（L2 regularization）
1. **正则化的工作原理**：
    
    - 通过惩罚大的参数值，使模型更简单
    - 鼓励模型使用所有特征，但每个特征的权重较小
    - 对于高阶多项式，会降低高次项的权重
如果模型过拟合，可以通过正则化来减少一些特征在模型中所占的权重，如：
$$
\min_{\theta} \dfrac {1}{2m} \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2 + \lambda \sum_{j=1}^n \theta_j^2
$$
即为$\theta$数据增添正则化参数$\lambda$（惩罚项）,它可以决定$\theta$参数的成本被膨胀的程度，减小$\lambda$,则可以减少过拟合，过于增大$\lambda$,则可能导致欠拟合
其他正则化方法：
- 早停法（Early Stopping）
- 参数范数惩罚（parameter norm penalties）
    - L1正则化
    - L2正则化
    - 最大范数正则化（max-norm regularization）
- 数据集增强（dataset augmentation）
- 噪声鲁棒性（Dropout等）
- 稀疏表示（sparse representation）
- ...