import numpy as np
from math import sqrt
from collections import Counter

#模拟KNN在clssiifier的代码运行过程
class KNNClassifier:

    def __init__(self, k):
        #初始化kNN分类器
        assert k >= 1, "k must be valid" #通过assert函数判断K是否大于1，判断合法
        self.k = k #返回K的成员变量
        self._X_train = None #_X设置私有变量，初始变量为None
        self._y_train = None #_y设置私有变量，初始变量为None

    def fit(self, X_train, y_train):
        #根据训练数据集X_train和y_train训练kNN分类器
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
                "must fit before predict!"#运行前应该已经运行过train，为None时报错
        assert X_predict.shape[1] == self._X_train.shape[1], \
                "the feature number of X_predict must be equal to X_train"#特征个数必须相同

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x): #计算过程 欧拉公式
        """给定单个待预测数据x，返回x的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x) ** 2))
                     for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k
