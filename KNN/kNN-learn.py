##编写的KNN算法
import numpy as np
from math import sqrt
from collections import Counter

#X_train是一个矩阵，代表所有相关因素
#y_train是一个判断值,代表不同类别，[0,0,0,0,0,0,1,1,1,1,1,2,2,2,2....]
def kNN_classify(k, X_train, y_train, x): #确定KNN算法中的标签，K值，训练值X, 训练值y（判断值）, 测试值x
    #断言
    assert 1 <= k <= X_train.shape[0], "k must be valid" #K值报错
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train" #训练值X中的数量必须与y值个数相等
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"#输入的测试值x的特征数量必须与训练值X中的特征数量相同

    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]#通过欧拉公式计算距离
    nearest = np.argsort(distances) #通过argsort函数，将距离从近到远进行排序，返回的值为数据的索引值

    topK_y = [y_train[i] for i in nearest[:k]] #将所有距离中靠前的K个值选出来，返回训练值y中的所有值
    votes = Counter(topK_y) #通过Counter函数，对于所有y中所有元素进行投票统计

    return votes.most_common(1)[0][0]#对投票统计y个数最多的，返回实际值
