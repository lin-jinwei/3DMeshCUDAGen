import numpy as np
from sklearn.cluster import KMeans

# 假设有N个三维空间点
N = 100  # 总点数
points = np.random.rand(N, 3)  # 随机生成N个三维空间点

print(f'{points = }')

# 设置聚类的数量为N/10
num_clusters = N // 10

# 创建KMeans实例
kmeans = KMeans(n_clusters=num_clusters)

# 执行K-means聚类
kmeans.fit(points)

# 获取聚类的质心
centroids = kmeans.cluster_centers_

# 获取每个点的簇标签
labels = kmeans.labels_

print("质心坐标：")
print(centroids)
print(f'{len(centroids) = }')
print("\n点的簇分配:")
print(labels)
print(f'{len(labels) = }')
