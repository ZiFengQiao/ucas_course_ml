import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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
def fit_gaussian_mixture_model(X, n_components=2, max_iter=100):
    # 合并数据
    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter)
    gmm.fit(X)
    return gmm.means_, gmm.covariances_, gmm.weights_

###请在此编写你的拟合函数！
def my_fit_GMM(X, max_iter=100):
    means = 0
    cov = 0
    weights = 0
    # 直接编写即可，不一定要使用结构体。
    return means, cov, weights


# 请改为用my_fit_GMM拟合散点分布。并输出估计的均值,方差和混合项系数。
means, cov, weights = fit_gaussian_mixture_model(X)
# means, cov, weights = my_fit_GMM(X) 

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