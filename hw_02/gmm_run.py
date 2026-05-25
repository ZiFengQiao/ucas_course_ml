import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        # 初始化高斯混合模型
        self.n_components = n_components  # 混合成分数量
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值

    def fit(self, X):
        # 适应数据以估计参数
        n_samples, n_features = X.shape

        # 初始化参数
        np.random.seed(42)
        self.weights = np.full(self.n_components, 1 / self.n_components)  # 初始化权重
        random_row = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_row]  # 随机选择初始均值
        self.covariances = np.array([np.cov(X, rowvar=False)] * self.n_components)  # 初始化协方差

        log_likelihood = 0
        for iteration in range(self.max_iter):
            # E-step: 计算责任
            responsibilities = self._e_step(X)

            # M-step: 更新参数
            self._m_step(X, responsibilities)

            # 检查收敛
            new_log_likelihood = self._compute_log_likelihood(X)
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

    def _e_step(self, X):
        # 计算样本属于每个成分的概率（责任）
        likelihood = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            likelihood[:, k] = self.weights[k] * multivariate_normal(
                self.means[k], self.covariances[k]
            ).pdf(X)
        total_likelihood = likelihood.sum(axis=1)[:, np.newaxis]
        responsibilities = likelihood / total_likelihood
        return responsibilities

    def _m_step(self, X, responsibilities):
        # 使用责任更新模型参数
        Nk = responsibilities.sum(axis=0)  # 每个成分的有效样本数
        self.weights = Nk / X.shape[0]  # 更新权重
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]  # 更新均值
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(
                responsibilities[:, k] * diff.T, diff
            ) / Nk[k]  # 更新协方差

    def _compute_log_likelihood(self, X):
        # 计算对数似然以检查收敛
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += self.weights[k] * multivariate_normal(
                self.means[k], self.covariances[k]
            ).pdf(X)
        return np.sum(np.log(log_likelihood))

    def predict(self, X):
        # 预测样本所属的成分
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

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

    plt.savefig("img.jpg")

# 示例用法
if __name__ == "__main__":
    # 生成一些数据
    np.random.seed(0)
    X = np.vstack([
        np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=100),
        np.random.multivariate_normal(mean=[5, 5], cov=[[1, 0], [0, 1]], size=100)
    ])

    # 创建并拟合GMM模型
    gmm = GMM(n_components=2)
    gmm.fit(X)
    draw_results(gmm.means, gmm.covariances, gmm.weights)
    # 打印参数
    print("Weights:", gmm.weights)
    print("Means:", gmm.means)
    print("Covariances:", gmm.covariances)

    # 预测
    labels = gmm.predict(X)
    print("Predicted labels:", labels)
