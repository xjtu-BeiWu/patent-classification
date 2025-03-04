# -*- coding: utf-8 -*-
# @Time : 7/22/19 3:50 PM
# @Author : Bei Wu
# @Site : 
# @File : pcaProcess.py
# @Software: PyCharm

import matplotlib.pyplot as plt  # 加载matplotlib用于数据的可视化
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA  # 加载PCA算法包
from sklearn.decomposition import SparsePCA

data = load_iris()
y = data.target
x = data.data
pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
spca = SparsePCA(n_components=2)
reduced_x = pca.fit_transform(x)  # 对样本进行降维
reduced_x_2 = spca.fit_transform(x)

red_x, red_y = [], []
# blue_y: List[Any]
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])

    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])

    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

# 可视化
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
