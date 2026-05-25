import numpy as np
import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture

# 运行这个脚本之前，请先确保安装有numpy, matplotlib, scikit-learn, scipy库。

# 初始化分布参数
MU1 = np.array([1, 2])
SIGMA1 = np.array([[1, 0], [0, 0.5]])
MU2 = np.array([-1, -1])
SIGMA2 = np.array([[1, 0], [0, 1]])

# 生成数据点
data1 = np.random.multivariate_normal(MU1, SIGMA1, 1000)
data2 = np.random.multivariate_normal(MU2, SIGMA2, 1000)
X = np.concatenate([data1, data2])

#对X进行随机打乱，此步复现时不可忽略
np.random.shuffle(X)

# 接下来绘制二维散点图
plt.scatter(data1[:, 0], data1[:, 1], label='Class 1')
plt.scatter(data2[:, 0], data2[:, 1], label='Class 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Data Points')
plt.legend()
plt.show()

# 请自己编程实现高斯混合模型的拟合办法。
# 示例中使用了scikit-learn提供的函数。自己实现时请勿调用此类已写好的高级函数。
# def fit_gaussian_mixture_model(X, n_components, max_iter=100):
#     # 合并数据
#     gmm = GaussianMixture(n_components=n_components, max_iter=max_iter)
#     gmm.fit(X)
#     return gmm.means_, gmm.covariances_, gmm.weights_

###请在此编写你的拟合函数！
def my_fit_GMM(X, max_iter=100):
    # 默认使用2个高斯分布进行拟合
    # 使用EM算法求解GMM参数

    n_samples, n_features = X.shape
    n_components = 2

    # ---------- 初始化参数 ----------
    # 用K-means策略初始化：随机选取2个样本作为初始均值
    rng = np.random.RandomState(42)
    indices = rng.choice(n_samples, n_components, replace=False)
    means = X[indices].copy().astype(np.float64)

    # 初始化协方差矩阵为单位矩阵
    covs = np.array([np.eye(n_features) * np.var(X) for _ in range(n_components)])

    # 初始化混合系数为均匀分布
    weights = np.ones(n_components) / n_components

    for iteration in range(max_iter):

        # -----------------------------------------------
        # E-Step: 计算责任度
        #         - 计算每个样本，来自不同高斯分布的概率密度
        #         - 计算思想：某个高斯分布生成该样本的概率密度 / 所有高斯分布生成该样本的概率密度之和
        #         - 协方差矩阵：在计算概率密度时，需要使用马氏距离，涉及协方差矩阵的逆，需要考虑求逆的数值稳定性问题。
        #         - 在一维的时候，使用(x-μ)^2 / σ^2来计算概率密度，实际上就是标准化后的距离，二维中对应的标准化距离是(x-μ)^T Σ^{-1} (x-μ)，其中Σ是协方差矩阵。
        
        
        
        # 计算每个样本在每个高斯分布下的概率密度
        # 使用协方差矩阵计算概率密度

        resp = np.zeros((n_samples, n_components))
        for k in range(n_components):
            # 添加小正则项防止协方差矩阵奇异，并计算协方差矩阵的逆
            cov_reg = covs[k] + 1e-6 * np.eye(n_features)
            cov_inv = np.linalg.inv(cov_reg)

            # 计算马氏距离下的标准化距离
            diff = X - means[k]
            mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)

            # 计算对数概率密度
            log_det = np.log(np.linalg.det(cov_reg))
            log_pdf = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)

            resp[:, k] = np.log(weights[k]) + log_pdf

        # 减去最大值防止数值下溢，再取指数
        log_resp_max = resp.max(axis=1, keepdims=True)
        resp = np.exp(resp - log_resp_max)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum = np.where(resp_sum < 1e-300, 1e-300, resp_sum)
        resp /= resp_sum  # shape: (n_samples, n_components), 每行和为1

        # -----------------------------------------------
        # M-Step: 根据责任度更新参数
        Nk = resp.sum(axis=0)

        # 更新混合系数
        weights = Nk / n_samples

        # 更新均值
        for k in range(n_components):
            means[k] = (resp[:, k:k+1] * X).sum(axis=0) / Nk[k]

        # 更新协方差矩阵
        for k in range(n_components):
            diff = X - means[k]
            covs[k] = (resp[:, k:k+1] * diff).T @ diff / Nk[k]
            # 保证协方差矩阵对称且正定
            covs[k] = (covs[k] + covs[k].T) / 2

    return means, covs, weights


# 请改为用my_fit_GMM拟合散点分布。并输出估计的均值,方差和混合项系数。
# means_ref, cov_ref, weights_ref = fit_gaussian_mixture_model(X, 2)
means, cov, weights = my_fit_GMM(X)

print("Data1.Means:")
print(means[0])
print("Data2.Means:")
print(means[1])

print("Data1.Covariances:")
print(cov[0])
print("Data2.Covariances:")
print(cov[1])

print("Weights:")
print(weights)

#这部分已经写好了绘制结果的函数，可用于验证你的算法效果
from scipy.stats import multivariate_normal

def draw_results(m,s,w):
    # 定义两个二维高斯分布的参数
    m1 = m[0]
    s1 = s[0]
    
    m2 = m[1]
    s2 = s[1]

    # 生成网格点
    x, y = np.mgrid[-5:5:.01, -5:5:.01]
    pos = np.dstack((x, y))

    # 计算每个点的概率密度值
    rv1 = multivariate_normal(m1, s1)
    rv2 = multivariate_normal(m2, s2)
    z1 = rv1.pdf(pos)
    z2 = rv2.pdf(pos)

    # 混合两个高斯分布的概率密度函数
    z = w[0] * z1 + w[1] * z2  # 设置混合比例

    # 绘制3D图
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability Density')
    ax.set_title('3D Plot of Mixture Gaussian Probability Density')

    plt.show()
    
draw_results(means, cov, weights)

#一切正常的话，就能看到拟合的可视化结果了