import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_dataset(test_size=0.3, random_state=42):
    """
    加载鸢尾花数据，并做简单离散化，便于使用ID3决策树
    返回：
        X_train, X_test, y_train, y_test, feature_names, target_names
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # 对每一列特征做离散化：按该列中位数划分为 0/1
    X_discrete = discretize_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_discrete, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names, target_names


def discretize_features(X):
    """
    将连续特征离散化。
    这里采用最简单的方法：每一列按中位数划分成 0/1
    """
    X_discrete = np.zeros_like(X, dtype=int)
    for j in range(X.shape[1]):
        median_val = np.median(X[:, j])
        X_discrete[:, j] = (X[:, j] > median_val).astype(int)
    return X_discrete


def calc_entropy(y):
    """
    计算标签集合 y 的信息熵
    """
    counter = Counter(y)
    total = len(y)
    entropy = 0.0

    for count in counter.values():
        p = count / total
        entropy -= p * np.log2(p)

    return entropy


def split_dataset(X, y, feature_index, feature_value):
    """
    按指定特征的指定取值划分数据集
    返回满足 X[:, feature_index] == feature_value 的子集
    同时删除该特征列，便于后续递归
    """
    sub_X = []
    sub_y = []

    for i in range(len(X)):
        if X[i][feature_index] == feature_value:
            reduced_feature = np.concatenate((X[i][:feature_index], X[i][feature_index + 1:]))
            sub_X.append(reduced_feature)
            sub_y.append(y[i])

    return np.array(sub_X), np.array(sub_y)


def calc_information_gain(X, y, feature_index):
    """
    计算某一特征的信息增益
    """
    base_entropy = calc_entropy(y)
    feature_values = set(X[:, feature_index])

    conditional_entropy = 0.0
    for value in feature_values:
        sub_X, sub_y = split_dataset(X, y, feature_index, value)
        prob = len(sub_y) / len(y)
        conditional_entropy += prob * calc_entropy(sub_y)

    info_gain = base_entropy - conditional_entropy
    return info_gain


def choose_best_feature(X, y):
    """
    选择信息增益最大的特征
    """
    num_features = X.shape[1]
    best_gain = -1
    best_feature = -1

    for i in range(num_features):
        gain = calc_information_gain(X, y, i)
        if gain > best_gain:
            best_gain = gain
            best_feature = i

    return best_feature


def majority_vote(y):
    """
    当无法继续划分时，返回出现次数最多的类别
    """
    counter = Counter(y)
    return counter.most_common(1)[0][0]


def build_tree(X, y, feature_names):
    """
    递归构造决策树
    参数:
        X: 当前数据集的特征
        y: 当前数据集的标签
        feature_names: 当前还可用的特征名列表

    返回:
        决策树（字典形式）或叶子节点类别
        
    树的表示形式：
    {
        'feature': 最优特征名,
        'feature_index': 最优特征索引,
        'children': {
            特征取值1: 子树,
            特征取值2: 子树,
            ...
        }
    }
    叶子节点直接返回类别标签
    """

    # =========================
    # 1. 递归终止条件
    # =========================

    # 情况1：如果当前所有样本都属于同一类别
    # 说明已经不需要继续划分了
    # 直接返回该类别，作为叶子节点
    if len(set(y)) == 1:
        return y[0]

    # 情况2：如果当前已经没有可用特征了
    # 无法再继续划分
    # 返回当前标签中数量最多的类别
    if not feature_names or X.shape[1] == 0:
        return majority_vote(y)

    # =========================
    # 2. 选择当前最优划分特征
    # =========================

    # 调用已经提供好的函数，找到最优特征的索引
    best_choice = choose_best_feature(X, y)

    # 根据索引取出对应的特征名
    best_choice_name = feature_names[best_choice]

    # =========================
    # 3. 创建当前节点
    # =========================

    # 用字典表示一棵树
    # 至少包含：
    # - 当前划分特征名称
    # - 子节点字典 children
    tree = {
        'feature': best_choice_name,
        'feature_index': best_choice,
        'children': {}
    }


    # =========================
    # 4. 获取当前最优特征的所有取值
    # =========================

    # 找到该特征在当前数据集中出现过的所有取值
    # 获取对应特征的值
    feature_values = set(X[:, best_choice])


    # =========================
    # 5. 生成新的特征名列表
    # =========================

    # 因为当前最优特征已经被使用
    # 所以后续递归构造子树时，不能再使用它
    # 需要把它从 feature_names 中删除
    new_feature_names = feature_names.copy()
    new_feature_names.remove(best_choice_name)  # 直接删除特征名

    # =========================
    # 6. 对每个取值分别构建子树
    # =========================

    for value in feature_values:
        # 6.1 根据当前特征取值划分数据集
        # 得到子数据 sub_X, sub_y
        # 注意：划分后通常该特征列也要被删掉
        sub_X, sub_y = split_dataset(X, y, best_choice, value)

        # 6.2 如果划分后的子集为空
        # 说明这个分支没有对应样本
        # 直接把该分支设为当前数据中的多数类
        if len(sub_y) == 0:
            subtree = majority_vote(y)
        else:
            # 递归时传入去掉当前最优特征后的特征名列表
            subtree = build_tree(sub_X, sub_y, new_feature_names)

        # 6.3 否则，递归调用 build_tree
        # 用子数据继续构造子树
        # 6.4 将得到的结果挂到当前节点的 children 上
        # 形式类似：
        # tree['children'][value] = ...
        tree['children'][value] = subtree


    # =========================
    # 7. 返回当前构造好的树
    # =========================
    return tree


def predict_one(tree, x, feature_names):
    """
    对单个样本进行预测
    x 必须与 feature_names 对应
    """
    # 叶子节点
    if not isinstance(tree, dict):
        return tree

    feature_name = tree['feature']
    feature_index = feature_names.index(feature_name)
    feature_value = x[feature_index]

    # 若测试集中出现训练时未见过的取值，做一个兜底处理
    if feature_value not in tree['children']:
        child_values = list(tree['children'].values())
        leaf_labels = [v for v in child_values if not isinstance(v, dict)]
        if len(leaf_labels) > 0:
            return Counter(leaf_labels).most_common(1)[0][0]
        else:
            return 0

    subtree = tree['children'][feature_value]

    # 递归时删除当前特征
    new_x = np.concatenate((x[:feature_index], x[feature_index + 1:]))
    new_feature_names = feature_names[:feature_index] + feature_names[feature_index + 1:]

    return predict_one(subtree, new_x, new_feature_names)


def predict(tree, X, feature_names):
    """
    对多个样本进行预测
    """
    y_pred = []
    for x in X:
        y_pred.append(predict_one(tree, x, feature_names))
    return np.array(y_pred)


def eval_tree(tree, X_test, y_test, feature_names):
    """
    评估决策树分类效果
    """
    y_pred = predict(tree, X_test, feature_names)
    accuracy = np.mean(y_pred == y_test)
    return accuracy


def print_tree(tree, indent=""):
    """
    简单打印决策树结构
    """
    if not isinstance(tree, dict):
        print(indent + "Leaf:", tree)
        return

    print(indent + f"[Feature] {tree['feature']}")
    for value, subtree in tree['children'].items():
        print(indent + f"  ├─ value = {value}")
        print_tree(subtree, indent + "  │   ")


def main():
    X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset()

    # 转成 list，方便后面按名字删除特征
    feature_names = list(feature_names)

    tree = build_tree(X_train, y_train, feature_names)

    print("构造好的决策树：")
    print_tree(tree)

    acc = eval_tree(tree, X_test, y_test, feature_names)
    print("\n测试集准确率：{:.4f}".format(acc))

    # 展示几个样本预测结果
    y_pred = predict(tree, X_test[:10], feature_names)
    print("\n前10个测试样本预测结果：")
    for i in range(10):
        print(f"样本{i}: 预测={target_names[y_pred[i]]}, 真实={target_names[y_test[i]]}")


if __name__ == "__main__":
    main()