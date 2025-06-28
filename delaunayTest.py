import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# 生成一些随机点
points = np.random.rand(20, 2)

# 创建 Delaunay 三角剖分对象
tri = Delaunay(points)

# 计算所有点之间的距离矩阵
distance_matrix = squareform(pdist(points))

# 设置距离阈值
distance_threshold = 0.5

# 过滤掉距离超过阈值的边
valid_edges = []
for simplex in tri.simplices:
    for i in range(3):
        edge = (simplex[i], simplex[(i + 1) % 3])
        if distance_matrix[edge] <= distance_threshold:
            valid_edges.append(edge)

# 将有效的边转换为唯一边
unique_edges = set()
for edge in valid_edges:
    edge = tuple(sorted(edge))
    unique_edges.add(edge)

# 可视化
plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='gray', linestyle='--')
for edge in unique_edges:
    plt.plot(points[list(edge), 0], points[list(edge), 1], 'b-')
plt.plot(points[:, 0], points[:, 1], 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Delaunay Triangulation with Distance Threshold')
plt.show()