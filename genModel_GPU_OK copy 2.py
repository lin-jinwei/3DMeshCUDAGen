import os
import numpy as np
from numba import cuda, njit, float32, uint8, int32
import time
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
import cv2
import trimesh
from PIL import Image
from io import BytesIO
import pyrender
from math import radians, cos, sin
import math




# ============================================================================
# ============================================================================
# Functions Definition

@cuda.jit
def assign_clusters(points, centroids, labels):
    idx = cuda.grid(1)
    if idx < points.shape[0]:
        min_dist = float('inf')
        min_idx = 0
        for i in range(centroids.shape[0]):
            dist = 0
            for j in range(points.shape[1]):
                temp = points[idx, j] - centroids[i, j]
                dist += temp * temp
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        labels[idx] = min_idx


# CUDA 核函数：更新聚类中心
@cuda.jit
def update_centroids(points, centroids, labels, counts):
    idx = cuda.grid(1)
    if idx < centroids.shape[0]:
        count = 0
        centroid = cuda.local.array(3, float32)
        for i in range(points.shape[0]):
            if labels[i] == idx:
                for j in range(3):
                    centroid[j] += points[i, j]
                count += 1
        for j in range(3):
            if count > 0:
                centroid[j] /= count
        for j in range(3):
            centroids[idx, j] = centroid[j]
        counts[idx] = count



def kmeans_numba(points, num_clusters, max_iter=10):
    # 初始化
    centroids = points[np.random.choice(points.shape[0], num_clusters, replace=False)]
    labels = np.zeros(points.shape[0], dtype=np.int32)
    counts = np.zeros(num_clusters, dtype=np.int32)

    # 将数据和标签传输到设备
    points_device = cuda.to_device(points)
    centroids_device = cuda.to_device(centroids)
    labels_device = cuda.to_device(labels)
    counts_device = cuda.to_device(counts)

    threadsperblock = 32
    blockspergrid = (points.shape[0] + (threadsperblock - 1)) // threadsperblock

    for _ in range(max_iter):
        assign_clusters[blockspergrid, threadsperblock](points_device, centroids_device, labels_device)
        update_centroids[blockspergrid, threadsperblock](points_device, centroids_device, labels_device, counts_device)
        centroids_device.copy_to_host(centroids)
        counts_device.copy_to_host(counts)

    return centroids


# def calculate_distance(p1, p2):
#     return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# 定义CUDA内核来计算点对之间的距离
@cuda.jit
def calculate_distances_kernel(points1, points2, distances):
    idx = cuda.grid(1)
    if idx < points1.shape[0]:
        # 计算单个点对之间的距离
        dx = points1[idx, 0] - points2[idx, 0]
        dy = points1[idx, 1] - points2[idx, 1]
        distances[idx] = (dx * dx + dy * dy) ** 0.5

def calculate_distances(points1, points2):
    # 确保输入是numpy数组并具有正确的类型
    points1 = np.asarray(points1, dtype=np.float32)
    points2 = np.asarray(points2, dtype=np.float32)

    # 创建结果数组
    distances = np.zeros(points1.shape[0], dtype=np.float32)

    # 配置线程块和线程网格大小
    threadsperblock = 256
    blockspergrid = (distances.size + (threadsperblock - 1)) // threadsperblock

    # 启动CUDA内核
    calculate_distances_kernel[blockspergrid, threadsperblock](points1, points2, distances)

    return distances


# CUDA内核用于计算所有点对之间的距离
@cuda.jit
def compute_distances_kernel(points, distances):
    i, j = cuda.grid(2)
    if i < points.shape[0] and j > i:
        dx = points[i, 0] - points[j, 0]
        dy = points[i, 1] - points[j, 1]
        distances[i, j] = (dx * dx + dy * dy) ** 0.5
        distances[j, i] = distances[i, j]  # 对称矩阵

# 函数用于找到最近的未使用的点对
def find_pairs(points):
    num_points = len(points)
    if num_points == 0 or num_points % 2 != 0:
        raise ValueError("The number of points must be even.")
    
    # 初始化距离矩阵
    distances = np.full((num_points, num_points), np.inf, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)

    # 配置CUDA网格和线程块大小
    threadsperblock = (16, 16)
    blockspergrid_x = (num_points + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (num_points + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # 启动CUDA内核计算所有点对之间的距离
    compute_distances_kernel[blockspergrid, threadsperblock](points, distances)

    # 将距离矩阵复制回主机内存进行后续处理
    distances = distances.copy_to_host() if isinstance(distances, cuda.devicearray.DeviceNDArray) else distances
    
    # 在CPU上完成配对逻辑
    used = [False] * num_points
    pairs = []

    for _ in range(num_points // 2):
        min_distance = np.inf
        pair = None
        
        for i in range(num_points):
            if not used[i]:
                for j in range(i + 1, num_points):
                    if not used[j] and distances[i, j] < min_distance:
                        min_distance = distances[i, j]
                        pair = (i, j)
        
        if pair is not None:
            i, j = pair
            used[i] = True
            used[j] = True
            pairs.append((points[i], points[j]))
    
    return pairs


def get_npy_data(file_path):
    file_path = os.path.join(script_directory, file_path)
    data = np.load(file_path)
    return data



# CUDA 内核用于转换灰度图并获取非零像素坐标
@cuda.jit
def convert_to_grey_and_get_nonzero(data_rgb, data_grey, nonzero_indices):
    x, y = cuda.grid(2)
    if x < data_rgb.shape[0] and y < data_rgb.shape[1]:
        # 转换为灰度值
        grey_value = (data_rgb[x, y, 0] * 0.299 + data_rgb[x, y, 1] * 0.587 + data_rgb[x, y, 2] * 0.114)
        data_grey[x, y] = grey_value
        
        # 如果灰度值大于0，则记录坐标
        if grey_value > 0:
            idx = cuda.atomic.add(nonzero_indices, 0, 1)
            if idx < nonzero_indices.size - 1:  # 确保不越界
                nonzero_indices[idx + 1] = x * data_rgb.shape[1] + y

# 封装成函数且函数名与函数参数不变
def get_coordinates(data):
    start_time = time.time()
    
    # 创建设备上的副本
    data_device = cuda.to_device(data)

    # 准备结果容器
    height, width, _ = data.shape
    data_grey_device = cuda.device_array((height, width), dtype=np.float32)
    max_nonzero = height * width
    nonzero_indices_device = cuda.device_array(max_nonzero + 1, dtype=np.int32)  # 第一个元素用于计数
    
    # 配置线程块和线程网格大小
    t_N = 32 * 10
    threadsperblock = (t_N, t_N)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # 启动 CUDA 内核
    convert_to_grey_and_get_nonzero[blockspergrid, threadsperblock](data_device, data_grey_device, nonzero_indices_device)

    # 获取结果
    data_grey = data_grey_device.copy_to_host().astype(np.uint8)
    nonzero_indices = nonzero_indices_device.copy_to_host()

    print(f"转换灰度图时间: {time.time() - start_time:.4f} 秒")

    # 构造 coordinates_x_y 和 coordinates_x_y_z 列表
    count = nonzero_indices[0]
    coordinates_x_y = []
    coordinates_x_y_z = []

    for i in range(1, count + 1):
        idx = nonzero_indices[i]
        x = idx // width
        y = idx % width
        z = data_grey[x, y]
        coordinates_x_y.append((x, y))
        coordinates_x_y_z.append([x, y, z])

    print(f"获取 coordinates_x_y 时间: {time.time() - start_time:.4f} 秒")
    print(f"获取 coordinates_x_y_z 时间: {time.time() - start_time:.4f} 秒")

    return coordinates_x_y, coordinates_x_y_z



@cuda.jit
def convert_to_greyscale(data_coordinates, data_grey):
    x, y = cuda.grid(2)
    if x < data_coordinates.shape[0] and y < data_coordinates.shape[1]:
        # 计算灰度值
        rgb = data_coordinates[x, y, :3]
        grey = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        data_grey[x, y] = np.uint8(grey)

@cuda.jit
def filter_and_extract_coordinates(data_grey, data_segs, coordinates_x_y_z, valid_indices, count):
    x, y = cuda.grid(2)
    if x < data_grey.shape[0] and y < data_grey.shape[1]:
        if data_grey[x, y] > 0:
            idx = cuda.atomic.add(count, 0, 1)
            coordinates_x_y_z[idx * 3] = x
            coordinates_x_y_z[idx * 3 + 1] = y
            coordinates_x_y_z[idx * 3 + 2] = data_grey[x, y]
            valid_indices[idx] = (x << 16) | y

def get_coordinates_segs(data_coordinates, data_segs):
    # 定义线程块的大小和网格的大小
    t_N = 32
    threadsperblock = (t_N, t_N)
    blockspergrid_x = (data_coordinates.shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (data_coordinates.shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # 分配GPU内存
    d_data_coordinates = cuda.to_device(data_coordinates)
    d_data_grey = cuda.device_array((data_coordinates.shape[0], data_coordinates.shape[1]), dtype=np.uint8)
    d_data_segs = cuda.to_device(data_segs)
    
    # 调用CUDA核函数进行灰度转换
    convert_to_greyscale[blockspergrid, threadsperblock](d_data_coordinates, d_data_grey)
    
    # 计算最大可能的有效点数量
    max_points = data_coordinates.shape[0] * data_coordinates.shape[1]
    d_coordinates_x_y_z = cuda.device_array(max_points * 3, dtype=np.int32)
    d_valid_indices = cuda.device_array(max_points, dtype=np.int32)
    d_count = cuda.to_device(np.array([0], dtype=np.int32))
    
    # 调用CUDA核函数进行条件筛选和坐标提取
    filter_and_extract_coordinates[blockspergrid, threadsperblock](d_data_grey, d_data_segs, d_coordinates_x_y_z, d_valid_indices, d_count)
    
    # 将结果从设备复制回主机
    count = d_count.copy_to_host()[0]
    coordinates_x_y_z = d_coordinates_x_y_z.copy_to_host()[:count * 3].reshape(-1, 3)
    valid_indices = d_valid_indices.copy_to_host()[:count]
    
    # 构建coordinates_x_y 和 Dict_x_y_z_segs
    coordinates_x_y = [(idx >> 16, idx & 0xFFFF) for idx in valid_indices]
    Dict_x_y_z_segs = {(x, y): [z, seg] for (x, y, z), seg in zip(coordinates_x_y_z, data_segs[valid_indices >> 16, valid_indices & 0xFFFF])}
    
    return coordinates_x_y, coordinates_x_y_z, Dict_x_y_z_segs



# CUDA 内核用于标记需要保留的行
@cuda.jit
def mark_rows_to_keep(coordinates_x_y_z, gap, keep_mask):
    idx = cuda.grid(1)
    if idx < coordinates_x_y_z.shape[0]:
        keep_mask[idx] = (idx % gap) != 0

# 封装成函数且函数名与函数参数不变
def interleave_delete(coordinates_x_y_z, gap):
    # 创建设备上的副本
    coordinates_device = cuda.to_device(np.array(coordinates_x_y_z, dtype=np.float32))
    
    # 准备结果容器和掩码容器
    n_rows = len(coordinates_x_y_z)
    keep_mask_device = cuda.device_array(n_rows, dtype=np.int32)

    # 配置线程块和线程网格大小
    threadsperblock = 256 * 2
    # blockspergrid = (n_rows + (threadsperblock - 1)) // threadsperblock
    blockspergrid = 32 * 100

    # 启动 CUDA 内核
    mark_rows_to_keep[blockspergrid, threadsperblock](coordinates_device, gap, keep_mask_device)

    # 获取掩码结果
    keep_mask = keep_mask_device.copy_to_host()

    # 构建新的坐标列表，只包含需要保留的行
    filtered_coordinates = [coordinates_x_y_z[i] for i in range(n_rows) if keep_mask[i]]

    return filtered_coordinates



@cuda.jit
def kmeans_cuda_kernel(coordinates_x_y_z, centroids, assignments):
    idx = cuda.grid(1)
    if idx < coordinates_x_y_z.shape[0]:
        # 计算距离
        min_dist = float32(1e10)
        min_centroid_idx = 0
        for i in range(centroids.shape[0]):
            dist = (coordinates_x_y_z[idx, 0] - centroids[i, 0]) ** 2 + \
                   (coordinates_x_y_z[idx, 1] - centroids[i, 1]) ** 2 + \
                   (coordinates_x_y_z[idx, 2] - centroids[i, 2]) ** 2
            if dist < min_dist:
                min_dist = dist
                min_centroid_idx = i
        assignments[idx] = min_centroid_idx

def kmeans_numba(coordinates_x_y_z, num_clusters, max_iter=100):
    # 初始化质心
    np.random.seed(0)
    indices = np.random.choice(coordinates_x_y_z.shape[0], num_clusters, replace=False)
    centroids = coordinates_x_y_z[indices].copy()  # 确保是一个副本

    # 初始化assignments
    assignments = np.zeros(coordinates_x_y_z.shape[0], dtype=np.int32)

    # 将数据转移到GPU
    d_coordinates = cuda.to_device(coordinates_x_y_z)
    d_centroids = cuda.to_device(centroids)
    d_assignments = cuda.to_device(assignments)

    for _ in range(max_iter):
        # 计算assignments
        threadsperblock = 256
        blockspergrid = (coordinates_x_y_z.shape[0] + (threadsperblock - 1)) // threadsperblock
        kmeans_cuda_kernel[blockspergrid, threadsperblock](d_coordinates, d_centroids, d_assignments)

        # 将assignments从设备复制回主机
        d_assignments.copy_to_host(assignments)

        # 更新质心
        new_centroids = np.zeros((num_clusters, 3), dtype=np.float32)
        counts = np.zeros(num_clusters, dtype=np.int32)
        for i in range(coordinates_x_y_z.shape[0]):
            new_centroids[assignments[i]] += coordinates_x_y_z[i]
            counts[assignments[i]] += 1
        for i in range(num_clusters):
            if counts[i] > 0:  # 避免除以零
                new_centroids[i] /= counts[i]

        # 检查收敛
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids



def centroids_cluster(coordinates_x_y_z, zoomN):
    num_clusters = int(len(coordinates_x_y_z) / zoomN)
    coordinates_x_y_z = np.array(coordinates_x_y_z, dtype=np.float32)
    centroids = kmeans_numba(coordinates_x_y_z, num_clusters)
    return centroids



# CUDA kernel for transforming points
@cuda.jit
def transform_points_kernel(points, points2, points_ori):
    idx = cuda.grid(1)
    if idx < points.shape[0]:
        x, y = points[idx]
        points2[idx, 0] = y
        points2[idx, 1] = -x
        points_ori[idx, 0] = points2[idx, 0]
        points_ori[idx, 1] = -points2[idx, 1]

def get_points_pointsOri(points):
    # Ensure input is a float32 numpy array and reshape it to (-1, 2) if necessary
    points = np.array(points, dtype=np.float32).reshape(-1, 2)

    # Allocate memory for output arrays
    points2 = np.empty_like(points, dtype=np.float32)
    points_ori = np.empty_like(points, dtype=np.float32)

    # Transfer data to the device
    d_points = cuda.to_device(points)
    d_points2 = cuda.to_device(points2)
    d_points_ori = cuda.to_device(points_ori)

    # Configure the blocks and grids
    threads_per_block = 256
    # blocks_per_grid = (points.shape[0] + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid = 32 * 100

    # Launch the CUDA kernel
    transform_points_kernel[blocks_per_grid, threads_per_block](d_points, d_points2, d_points_ori)

    # Copy result back to host
    points2 = d_points2.copy_to_host()
    points_ori = d_points_ori.copy_to_host()

    return points2, points_ori



@cuda.jit
def compute_distances(points, distances):
    i, j = cuda.grid(2)
    if i < points.shape[0] and j < points.shape[0]:
        dx = points[i, 0] - points[j, 0]
        dy = points[i, 1] - points[j, 1]
        distances[i, j] = math.sqrt(dx * dx + dy * dy)

def distance_matrix_Delaunay(points):
    # 使用scipy.spatial.Delaunay进行Delaunay三角剖分
    tri = Delaunay(points)
    
    # 计算所有点之间的距离矩阵
    n_points = points.shape[0]
    distances = np.zeros((n_points, n_points), dtype=np.float32)
    
    threadsperblock = (16, 16)
    blockspergrid_x = (distances.shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (distances.shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    d_points = cuda.to_device(points.astype(np.float32))
    d_distances = cuda.device_array_like(distances)
    
    compute_distances[blockspergrid, threadsperblock](d_points, d_distances)
    
    distances = d_distances.copy_to_host()
    
    return tri, distances



def get_valid_edges_simplex(tri, distance_matrix, distance_threshold):
    # 过滤掉距离超过阈值的边
    # tri.simplices 属性返回一个二维数组，
    # 其中每一行表示一个三角形，每一列表示该三角形的一个顶点的索引。
    valid_edges = []
    valid_simplex = []
    
    for simplex in tri.simplices:
        for i in range(3):
            edge = (simplex[i], simplex[(i + 1) % 3])
            if distance_matrix[edge] <= distance_threshold:
                valid_edges.append(edge)
                simplex_L = simplex.tolist()
                if simplex_L not in valid_simplex:
                    valid_simplex.append(simplex_L)
                    
    return valid_edges, valid_simplex



def draw_save_plt(plt_path, figsize, valid_edges, points, if_save, if_show):
    fig = plt.figure(figsize=figsize)
    # 获取图形的大小（以英寸为单位）
    fig_size_inches = fig.get_size_inches()
    # print(f"Figure size in inches: {fig_size_inches}")
    # 获取图形的 DPI
    dpi = fig.get_dpi()
    # print(f"Figure DPI: {dpi}")
    # 计算图形的大小（以像素为单位）
    fig_size_pixels = fig_size_inches * dpi
    # print(f"Figure size in pixels: {fig_size_pixels}")

    for edge in valid_edges:
        start_point = points[list(edge), 0]
        end_point = points[list(edge), 1]
        plt.plot(start_point, end_point, 'b-')

    plt.plot(points[:, 0], points[:, 1], 'ro', markersize=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Delaunay Image')
    plt.axis('equal')
    # 设置等比例轴
    if if_save:
        # 保存图像，保持原长宽比例
        plt.savefig(plt_path, bbox_inches='tight', pad_inches=0)
    if if_show:
        # 显示图像
        plt.show()
    return fig



@cuda.jit
def divide_segment_kernel(x1, y1, x2, y2, n, points):
    idx = cuda.grid(1)
    if idx <= n:
        dx = (x2 - x1) / n
        dy = (y2 - y1) / n
        new_x = x1 + idx * dx
        new_y = y1 + idx * dy
        points[idx * 2] = new_x
        points[idx * 2 + 1] = new_y

def divide_segment(x1, y1, x2, y2, n):
    if n <= 0:
        raise ValueError("n 必须是一个正整数")
    # 创建结果数组
    points = np.zeros((n + 1) * 2, dtype=np.float32)
    
    # 将数据传输到设备
    d_points = cuda.to_device(points)
    
    # 配置CUDA内核
    threadsperblock = 256
    blockspergrid = (n + 1 + (threadsperblock - 1)) // threadsperblock
    
    # 执行CUDA内核
    divide_segment_kernel[blockspergrid, threadsperblock](x1, y1, x2, y2, n, d_points)
    
    # 将结果从设备复制回主机
    points = d_points.copy_to_host()
    
    # 转换为元组列表
    points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
    return points



def showTime(start_time, str):
    now_time = time.time()
    elapsed_time = now_time - start_time
    print('\n' + str + f"========> : {elapsed_time:.2f} 秒")
    return elapsed_time
    

def showTime_remove(start_time, remove_time1, remove_time2, str):
    now_time = time.time()
    elapsed_time = now_time - start_time - abs(remove_time1-remove_time2)
    print('\n' + str + f"========> : {elapsed_time:.2f} 秒")
    return elapsed_time


def get_rgb_value(image, x, y):
    # OpenCV 默认以 BGR 格式读取图片，转换为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 获取 (x, y) 位置的 RGB 值
    try:
        (r, g, b) = image_rgb[y, x]
    except IndexError:
        print("提供的坐标超出图片范围")
        return
    return (r, g, b) 



def get_Z_from_points(x, y, coordinates_x_y_z):
    for x_y_z in coordinates_x_y_z:
        x_3 = x_y_z[0]
        y_3 = x_y_z[1]
        if x == x_3 and y == y_3:
            return x_y_z[2]



def draw_save_cv2(data, 
                  coordinates_x_y_z,
                  points_ori,
                  valid_edges,
                  zoomWH, 
                  color_line, 
                  thickness_line, 
                  zoom_Z, 
                  criterion,
                  if_save,
                  if_show,
                  img_cv2_save_path,
                  output_file_path,
                  radius_points,
                  color_points,
                  thickness_points,
                  start_time):
    
    pre_W = data.shape[1]
    pre_H = data.shape[0]
    pic_W = int(pre_W / zoomWH)
    pic_H = int(pre_H / zoomWH)

    # 在图片上绘制每个点
    image = cv2.imread(output_file_path)

    new_valid_points = []
    valid_x_y_z_edges = []
    # 绘制多条线段
    for edge in valid_edges:
        x_points = points_ori[list(edge), 0]
        y_points = points_ori[list(edge), 1]
        
        z1 = get_Z_from_points(y_points[0], x_points[0], coordinates_x_y_z)
        z2 = get_Z_from_points(y_points[1], x_points[1], coordinates_x_y_z)
        [x1, x2] = np.round(x_points).astype(int)
        [y1, y2] = np.round(y_points).astype(int)
        z1 = z1 * zoom_Z
        z2 = z2 * zoom_Z
        
        valid_x_y_z_0 = [y_points[0], x_points[0], z1]
        valid_x_y_z_1 = [y_points[1], x_points[1], z2]
        
        outLine = False
        outColors = 0 

        start_point = [x1, y1, z1]
        end_point = [x2, y2, z2]
       
        # 获取线段中点的坐标值
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        (r, g, b) = get_rgb_value(image, x, y)
        
        # cv2.circle(image, (x, y), 10, (255, 0, 0), thickness_points)
        
        if (r, g, b) != color_line:
            # print(f'{(r, g, b) = }')
            for c in (r, g, b):
                if c <= criterion:
                    outColors += 1
            if outColors == 3:
                outLine = True
        else:
            outLine = True

        if not outLine:
            # valid_x_y_z_edges.append([start_point, end_point])
            valid_x_y_z_edges.append([valid_x_y_z_0, valid_x_y_z_1])
            point_start_in = False
            point_end_in = False
            for it in new_valid_points:
                if np.array_equal(it, start_point):
                    point_start_in = True
                if np.array_equal(it, end_point):
                    point_end_in = True
            if not point_start_in:
                new_valid_points.append(start_point)
            if not point_end_in:
                new_valid_points.append(end_point)
            cv2.line(image, start_point[0:2], end_point[0:2], color_line, thickness_line)

    elapsed_time = showTime(start_time, '显示图像-2: CV2绘制-线-时间')
    para_draw_lines_CV2_T = elapsed_time
    # print(f'{len(new_valid_points) = }')
    # print(f'{len(valid_x_y_z_edges) = }')

    # 绘制多个点
    for point in new_valid_points:
        x, y = point[0], point[1]
        cv2.circle(image, (x, y), radius_points, color_points, thickness_points)

    elapsed_time = showTime(start_time, '显示图像-3: CV2绘制-点-时间')
    para_draw_points_CV2_T = elapsed_time
    
    # 使用PNG格式保存图像，并设置压缩级别（0-9，0表示无压缩）
    if if_save:
        cv2.imwrite(img_cv2_save_path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        elapsed_time = showTime(start_time, '显示图像-4: CV2保存图片-时间')
        para_save_CV2_T = elapsed_time

    if if_show:
        window_name = "CV2 Image"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, pic_W, pic_H)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        showTime(start_time, '显示图像-5: CV2展示图片-时间')
        
    T_L = [
        para_draw_lines_CV2_T,
        para_draw_points_CV2_T,
        para_save_CV2_T
    ]
        
    return new_valid_points, valid_x_y_z_edges, image, T_L
        
    
    
def save_show_obj(obj_path, coordinates_points, valid_simplex_faces, if_show):
    # 创建一个Trimesh对象
    mesh = trimesh.Trimesh(vertices=coordinates_points, faces=valid_simplex_faces)
    # 可视化模型
    if if_show:
        mesh.show()
    # 保存模型为OBJ文件
    mesh.export(obj_path)
    return mesh



def get_points_tri_valid_edges_simplex(coordinates_x_y_z, distance_threshold=100):
    coordinates_x_y = coordinates_x_y_z[:, :2]
    points, points_ori = get_points_pointsOri(coordinates_x_y)
    # 使用scipy.spatial.Delaunay进行Delaunay三角剖分
    tri, distance_matrix = distance_matrix_Delaunay(points)
    # 获取有效的边与顶点索引
    # 设置距离阈值
    valid_edges, valid_simplex = get_valid_edges_simplex(tri, distance_matrix, distance_threshold)
    return points, points_ori, valid_edges, valid_simplex



@cuda.jit(parallel=True)
def parallel_sort(arr):
    return np.sort(arr)

@cuda.jit
def check_duplicates(sorted_arr, result):
    idx = cuda.grid(1)
    if idx > 0 and idx < sorted_arr.shape[0]:
        if (sorted_arr[idx] == sorted_arr[idx - 1]).all():
            cuda.atomic.max(result, 0, 1)

def has_duplicates(list_A):
    # 确保输入是 NumPy 数组
    list_A = np.array(list_A, dtype=np.float32)
    
    # 检查输入是否为空
    if list_A.size == 0:
        return False
    
    # 将每个子列表转换为元组以确保其可哈希
    try:
        tuples = np.array([tuple(item) for item in list_A], dtype=object)
    except TypeError:
        print("无法将所有元素转换为元组。")
        return False
    
    # 将元组转换为浮点数数组以便排序
    flat_tuples = np.array(tuples.tolist(), dtype=np.float32).reshape(-1, list_A.shape[1])
    
    # 排序数组
    sorted_flat_tuples = parallel_sort(flat_tuples)
    
    # 创建结果数组
    result = np.zeros(1, dtype=np.int32)
    
    # 配置 CUDA 内核
    threads_per_block = 256
    blocks_per_grid = (sorted_flat_tuples.shape[0] + threads_per_block - 1) // threads_per_block
    
    # 启动 CUDA 内核检查重复项
    check_duplicates[blocks_per_grid, threads_per_block](sorted_flat_tuples, result)
    
    # 检查结果
    return result[0] == 1




@cuda.jit
def swap_xy_kernel(input_array, output_array):
    idx = cuda.grid(1)
    if idx < input_array.shape[0]:
        x1 = input_array[idx, 0]
        y1 = input_array[idx, 1]
        z1 = input_array[idx, 2]
        output_array[idx, 0] = y1
        output_array[idx, 1] = x1
        output_array[idx, 2] = z1

def swap_x_y(input_list):
    if not isinstance(input_list, list) or len(input_list) != 3:
        raise ValueError("Input must be a list of three elements.")
    
    # 将输入列表转换为 NumPy 数组
    input_array = np.array(input_list, dtype=np.float32).reshape(1, 3)
    
    # 创建输出数组
    output_array = np.empty_like(input_array)
    
    # 配置 CUDA 内核
    threads_per_block = 256
    blocks_per_grid = (input_array.shape[0] + threads_per_block - 1) // threads_per_block
    
    # 启动 CUDA 内核
    swap_xy_kernel[blocks_per_grid, threads_per_block](input_array, output_array)
    
    # 返回结果
    return output_array.flatten().tolist()



def list_to_np_array(input_list, dtype=None):
    return np.array(input_list, dtype=dtype)




@cuda.jit
def check_simplex_kernel(valid_simplex, coordinates_x_y_z, valid_x_y_z_edges, flags):
    idx = cuda.grid(1)
    if idx < valid_simplex.shape[0]:
        a_b_c = valid_simplex[idx]
        pa = coordinates_x_y_z[a_b_c[0]]
        pb = coordinates_x_y_z[a_b_c[1]]
        pc = coordinates_x_y_z[a_b_c[2]]
        
        pa_x_y = (pa[0], pa[1])
        pb_x_y = (pb[0], pb[1])
        pc_x_y = (pc[0], pc[1])
        
        flag_connected_a_b = False
        flag_connected_a_c = False
        flag_connected_b_c = False
        
        for edge in valid_x_y_z_edges:
            x_y_edges_0 = (edge[0][0], edge[0][1])
            x_y_edges_1 = (edge[1][0], edge[1][1])
            
            # 检查 (pa_x_y == x_y_edges_0) and (pb_x_y == x_y_edges_1)
            if (pa_x_y[0] == x_y_edges_0[0]) and (pa_x_y[1] == x_y_edges_0[1]) and \
               (pb_x_y[0] == x_y_edges_1[0]) and (pb_x_y[1] == x_y_edges_1[1]):
                flag_connected_a_b = True
            
            # 检查 (pa_x_y == x_y_edges_1) and (pb_x_y == x_y_edges_0)
            elif (pa_x_y[0] == x_y_edges_1[0]) and (pa_x_y[1] == x_y_edges_1[1]) and \
                 (pb_x_y[0] == x_y_edges_0[0]) and (pb_x_y[1] == x_y_edges_0[1]):
                flag_connected_a_b = True
            
            # 检查 (pa_x_y == x_y_edges_0) and (pc_x_y == x_y_edges_1)
            if (pa_x_y[0] == x_y_edges_0[0]) and (pa_x_y[1] == x_y_edges_0[1]) and \
               (pc_x_y[0] == x_y_edges_1[0]) and (pc_x_y[1] == x_y_edges_1[1]):
                flag_connected_a_c = True
            
            # 检查 (pa_x_y == x_y_edges_1) and (pc_x_y == x_y_edges_0)
            elif (pa_x_y[0] == x_y_edges_1[0]) and (pa_x_y[1] == x_y_edges_1[1]) and \
                 (pc_x_y[0] == x_y_edges_0[0]) and (pc_x_y[1] == x_y_edges_0[1]):
                flag_connected_a_c = True
            
            # 检查 (pb_x_y == x_y_edges_0) and (pc_x_y == x_y_edges_1)
            if (pb_x_y[0] == x_y_edges_0[0]) and (pb_x_y[1] == x_y_edges_0[1]) and \
               (pc_x_y[0] == x_y_edges_1[0]) and (pc_x_y[1] == x_y_edges_1[1]):
                flag_connected_b_c = True
            
            # 检查 (pb_x_y == x_y_edges_1) and (pc_x_y == x_y_edges_0)
            elif (pb_x_y[0] == x_y_edges_1[0]) and (pb_x_y[1] == x_y_edges_1[1]) and \
                 (pc_x_y[0] == x_y_edges_0[0]) and (pc_x_y[1] == x_y_edges_0[1]):
                flag_connected_b_c = True
        
        if flag_connected_a_b and flag_connected_a_c and flag_connected_b_c:
            flags[idx] = 1

def get_valid_simplex_in_xyz(valid_simplex, coordinates_x_y_z, valid_x_y_z_edges):
    # 确保输入是 NumPy 数组
    valid_simplex = np.array(valid_simplex, dtype=np.int32)
    coordinates_x_y_z = np.array(coordinates_x_y_z, dtype=np.float32)
    valid_x_y_z_edges = np.array(valid_x_y_z_edges, dtype=np.float32)
    
    # 创建标志数组
    flags = np.zeros(valid_simplex.shape[0], dtype=np.int32)
    
    # 分配 GPU 数组
    d_valid_simplex = cuda.to_device(valid_simplex)
    d_coordinates_x_y_z = cuda.to_device(coordinates_x_y_z)
    d_valid_x_y_z_edges = cuda.to_device(valid_x_y_z_edges)
    d_flags = cuda.to_device(flags)
    
    # 配置 CUDA 内核
    threads_per_block = 256
    # blocks_per_grid = (valid_simplex.shape[0] + threads_per_block - 1) // threads_per_block
    blocks_per_grid = 32 * valid_simplex.shape[0] * threads_per_block
    
    # 启动 CUDA 内核
    check_simplex_kernel[blocks_per_grid, threads_per_block](
        d_valid_simplex, d_coordinates_x_y_z, d_valid_x_y_z_edges, d_flags
    )
    
    # 将结果复制回主机
    flags = d_flags.copy_to_host()
    
    # 获取有效的面
    valid_indices = np.where(flags == 1)[0]
    valid_simplex_faces = valid_simplex[valid_indices]
    
    return valid_simplex_faces.tolist()


def rotate_point_cloud(points, axis='z', angle_degrees=-90):
    # Convert the angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    
    # Define rotation matrices for each axis
    if axis.lower() == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis.lower() == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis.lower() == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
    
    # Apply the rotation matrix to all points
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points



@cuda.jit
def rotate_points_kernel(points, rotation_matrix, rotated_points):
    idx = cuda.grid(1)
    if idx < points.shape[0]:
        x = points[idx, 0]
        y = points[idx, 1]
        z = points[idx, 2]
        
        rx = rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y + rotation_matrix[0, 2] * z
        ry = rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y + rotation_matrix[1, 2] * z
        rz = rotation_matrix[2, 0] * x + rotation_matrix[2, 1] * y + rotation_matrix[2, 2] * z
        
        rotated_points[idx, 0] = rx
        rotated_points[idx, 1] = ry
        rotated_points[idx, 2] = rz

def rotate_point_cloud2(points, axis='z', angle_degrees=-90):
    # Convert the angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    
    # Define rotation matrices for each axis
    if axis.lower() == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis.lower() == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis.lower() == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
    
    # Ensure inputs are NumPy arrays
    points = np.array(points, dtype=np.float32)
    rotation_matrix = np.array(rotation_matrix, dtype=np.float32)
    
    # Allocate device memory
    d_points = cuda.to_device(points)
    d_rotation_matrix = cuda.to_device(rotation_matrix)
    d_rotated_points = cuda.device_array_like(d_points)
    
    # Configure CUDA kernel
    threads_per_block = 256 * 2
    # blocks_per_grid = (points.shape[0] + threads_per_block - 1) // threads_per_block
    blocks_per_grid = 32 * 10000000
    
    # Launch CUDA kernel
    rotate_points_kernel[blocks_per_grid, threads_per_block](d_points, d_rotation_matrix, d_rotated_points)
    
    # Copy results back to host
    rotated_points = d_rotated_points.copy_to_host()
    
    return rotated_points.tolist()



def int_to_padded_string(number, n):
    # Ensure that the number is non-negative and n is positive
    if number < 0:
        raise ValueError("The number must be non-negative.")
    if n <= 0:
        raise ValueError("The target length n must be positive.")
    # Convert the number to a string and pad it with zeros to the left
    padded_string = f"{number:0{n}d}"
    return padded_string


def plt_to_cv2(plt_figure):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    pil_image = Image.open(buf).convert("RGB")
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    buf.close()
    return cv2_image


def black_to_white(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # Create a mask where black (or near-black) pixels are white and others are black
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # Invert the mask so that non-black areas are preserved when we copy
    mask_inv = cv2.bitwise_not(mask)
    # Create a white background image
    white_bg = np.full_like(image, 255) if len(image.shape) == 3 else np.full_like(gray, 255)
    # Copy the original image onto the white background using the mask
    result = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
    result = cv2.add(result, cv2.bitwise_and(image, image, mask=mask))
    return result


def create_material(color):
    """Create a material with the specified base color, converting RGB (0-255) to (0-1)."""
    # Convert RGB values from 0-255 range to 0-1 range for pyrender
    color_normalized = [c / 255.0 for c in color]
    return pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=color_normalized + [1.0]  # Add alpha channel for opacity
    )



def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = cos(radians(theta) / 2.0)
    b, c, d = -axis * sin(radians(theta) / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), 0],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), 0],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, 0],
                     [0, 0, 0, 1]])


def apply_rotation(pose, rotation_x, rotation_y, rotation_z):
    """Apply rotations around X, Y, and Z axes to a pose matrix."""
    rotation_x_mat = rotation_matrix([1, 0, 0], rotation_x)
    rotation_y_mat = rotation_matrix([0, 1, 0], rotation_y)
    rotation_z_mat = rotation_matrix([0, 0, 1], rotation_z)
    rotation = rotation_x_mat @ rotation_y_mat @ rotation_z_mat
    return rotation @ pose



def render_front_view_to_cv2(
    mesh,
    camera_distance=1600,
    camera_left_offset=1000,
    camera_right_offset=0,
    camera_up_offset=-1800,
    camera_down_offset=0,
    camera_forward_offset=0,
    camera_backward_offset=0,
    field_of_view=100,
    render_width=600,
    render_height=1150,
    light_position_x=0,
    light_position_y=-1000,
    light_position_z=500,
    light_intensity=3,
    mesh_color=[166, 233, 166],
    camera_rotation_x=0,         # Rotation around X-axis in degrees
    camera_rotation_y=0,         # Rotation around Y-axis in degrees
    camera_rotation_z=0,         # Rotation around Z-axis in degrees
    light_rotation_x=-82,        # Rotation around X-axis in degrees for light
    light_rotation_y=0,          # Rotation around Y-axis in degrees for light
    light_rotation_z=0,          # Rotation around Z-axis in degrees for light
    if_show=False
):
    # Create materials for solid rendering
    solid_material = create_material(mesh_color)

    # Create a PyRender Mesh from the Trimesh mesh with the specified material for solid rendering
    solid_mesh = pyrender.Mesh.from_trimesh(mesh, material=solid_material, smooth=False)

    # Create a PyRender scene and add the solid mesh to it
    scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02])  # Set a low ambient light

    scene.add(solid_mesh)

    # Convert field of view from degrees to radians for the PerspectiveCamera
    yfov_radians = np.deg2rad(field_of_view)

    # Define camera parameters for a perspective (front) view with specified field of view
    camera = pyrender.PerspectiveCamera(yfov=yfov_radians)

    # Place the camera in front of the object looking down the -z axis at a distance specified by camera_distance.
    # Adjust position based on offsets and apply rotation
    camera_pose = np.array([
        [1, 0, 0, camera_left_offset - camera_right_offset],
        [0, 1, 0, camera_up_offset - camera_down_offset],
        [0, 0, 1, camera_distance + camera_forward_offset - camera_backward_offset],
        [0, 0, 0, 1]
    ])
    camera_pose = apply_rotation(camera_pose, camera_rotation_x, camera_rotation_y, camera_rotation_z)

    scene.add(camera, pose=camera_pose)

    # Add a directional light to illuminate the scene
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)

    # Position the light relative to the camera position
    light_pose = np.array([
        [1, 0, 0, light_position_x],
        [0, 1, 0, light_position_y],
        [0, 0, 1, light_position_z],
        [0, 0, 0, 1]
    ])
    light_pose = apply_rotation(light_pose, light_rotation_x, light_rotation_y, light_rotation_z)

    scene.add(light, pose=light_pose)

    # Set up the renderer with specified dimensions
    r = pyrender.OffscreenRenderer(render_width, render_height)

    # Render the scene
    color, depth = r.render(scene)

    # Convert RGB image to BGR for OpenCV
    bgr_image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    # Optionally display the image using OpenCV
    if if_show:
        cv2.imshow('3D Front View', bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Release resources
    r.delete()
    return bgr_image



def draw_transparent_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2, background_color=(0, 0, 0), alpha=0.6):
    # 获取文本尺寸
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 计算矩形坐标
    rect_start = (position[0] - 5, position[1] + baseline + 5)  # 矩形左下角
    rect_end = (position[0] + text_width + 5, position[1] - text_height - 5)  # 矩形右上角

    # 创建图像的副本以避免修改原始图像
    overlay = image.copy()

    # 绘制半透明矩形背景
    cv2.rectangle(overlay, rect_start, rect_end, background_color, -1)  # -1 表示填充矩形

    # 将半透明矩形应用到原图上
    output_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 在原图上绘制文本
    cv2.putText(output_image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    return output_image



def combined_cv2_imgs(combined_img_path, source_img, plt_fig, cv2_image, data_depth, mesh_cv2_img_has_redundant, mesh_cv2_img_remove_redundant, show_combined_cv2, save_combined_cv2, text):
    # 将plt转为cv2
    plt_fig_cv2 = plt_to_cv2(plt_fig)
    # 关闭绘图以释放内存
    plt.close(plt_fig)  
    
    cv2_image = black_to_white(cv2_image) 
    data_depth = black_to_white(data_depth)

    # 确保两个图像有相同的尺寸，如果需要的话调整大小
    if plt_fig_cv2.shape != cv2_image.shape:
        plt_fig_cv2 = cv2.resize(plt_fig_cv2, (cv2_image.shape[1], cv2_image.shape[0]))

    if source_img.shape[1] != cv2_image.shape[1]:
        source_img = cv2.resize(source_img, (source_img.shape[1], cv2_image.shape[0]))
        
    if mesh_cv2_img_has_redundant.shape[1] != cv2_image.shape[1]:
        mesh_cv2_img_has_redundant = cv2.resize(mesh_cv2_img_has_redundant, (cv2_image.shape[1], cv2_image.shape[0]))
        
    if mesh_cv2_img_remove_redundant.shape[1] != cv2_image.shape[1]:
        mesh_cv2_img_remove_redundant = cv2.resize(mesh_cv2_img_remove_redundant, (cv2_image.shape[1], cv2_image.shape[0]))

    combined_img = cv2.hconcat([source_img, data_depth, plt_fig_cv2, cv2_image, mesh_cv2_img_has_redundant, mesh_cv2_img_remove_redundant])
    
    # print(f'{cv2_image.shape[1] = }')
    
    # 定义文本参数
    position = (30, cv2_image.shape[0] - 80)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.9
    color = (255, 255, 255)  # 文本颜色
    thickness = 10
    background_color = (0, 0, 0)  # 黑色背景
    alpha = 0.6  # 透明度

    # 添加文本到图像
    combined_img = draw_transparent_text(combined_img, text, position, font, font_scale, color, thickness, background_color, alpha)

    # 显示结果图像
    if show_combined_cv2:
        # 显示拼接后的图像
        cv2.imshow('Combined Images', combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_combined_cv2:
        # 保存拼接图像
        cv2.imwrite(combined_img_path, combined_img)
        
    
   
        

# ============================================================================
# ============================================================================
# Implementation Code


# ============================
# Configuration

imge_ID = 1
para_imge_ID = imge_ID


# 聚类

# zoomN = 10000
# zoomN = 5000
# zoomN = 2500
# zoomN = 2000
# zoomN = 1500

zoomN = 1000
# zoomN = 500
# zoomN = 200
# zoomN = 100






# zoomN = 1176393
para_zoomN = zoomN
# ============================


start_time = time.time()
# print(f'{start_time = }')
script_path = os.path.abspath(__file__)
print("脚本文件路径是:", script_path)
script_directory = os.path.dirname(script_path)
print("脚本所在目录是:", script_directory)

imag_dir = './Outputs/reel1/sapiens_1b/'
imag_name = int_to_padded_string(imge_ID, 6) + '.jpg'
file_path_source = os.path.join(script_directory, imag_dir, imag_name)
source_img = cv2.imread(file_path_source)

file_path_coordinates = 'Outputs/reel1/sapiens_2b/depth_test/processed/' + int_to_padded_string(imge_ID, 6) + '.npy'
file_path_segs = 'Outputs/reel1/sapiens_1b/' + int_to_padded_string(imge_ID, 6) + '_seg.npy'
file_path_img = os.path.join(script_directory, file_path_coordinates)

imag_dir = '../model/'
imag_name = 'read_depth_npy.jpg'
output_file_path = os.path.join(script_directory, imag_dir, imag_name)

data_depth = np.load(file_path_img)
cv2.imwrite(output_file_path, data_depth)

data_coordinates = get_npy_data(file_path_coordinates)
elapsed_time = showTime(start_time, '加载 npy-data_coordinates 文件时间')
para_read_data_coordinates_T = elapsed_time
# print(f'{data_coordinates[0][0] = }')
# print(f'{data_coordinates.shape = }')

data_segs = get_npy_data(file_path_segs)
elapsed_time = showTime(start_time, '加载 npy-data_segs 文件时间')
para_read_data_segs_T = elapsed_time

# print(f'{data_segs[0][0] = }')
# print(f'{data_segs.shape = }')

# coordinates_x_y, coordinates_x_y_z = get_coordinates(data_coordinates)
coordinates_x_y, coordinates_x_y_z, Dict_x_y_z_segs = get_coordinates_segs(data_coordinates, data_segs)
# print(f'{len(coordinates_x_y_z) = }')
para_pre_points_N = len(coordinates_x_y_z)

# ==============================================================================
# 栅格删除法
# coordinates_x_y_z = interleave_delete(coordinates_x_y_z, 2)
# para_interleave_delete_points_N = len(coordinates_x_y_z)
# elapsed_time = showTime(start_time, '栅格删除法-获取 coordinates_x_y_z 时间')
# para_interleave_delete_T = elapsed_time
# ==============================================================================

coordinates_x_y_z = centroids_cluster(coordinates_x_y_z, zoomN)
elapsed_time = showTime(start_time, 'Numba-CUDA 聚类时间')
para_Numba_CUDA_T = elapsed_time

points, points_ori, valid_edges, valid_simplex = get_points_tri_valid_edges_simplex(coordinates_x_y_z)

# =====================================================================================
# 可视化

plt_path = 'model/delaunay_triangulation.png'
figsize = (8, 16)
if_save = True
# if_show = True
if_show = False
plt_fig = draw_save_plt(plt_path, figsize, valid_edges, points, if_save, if_show)
elapsed_time = showTime(start_time, '显示图像-1: 原始点线图-时间')
para_show_plt_fig_T = elapsed_time

zoomWH = 3
# 设置线的颜色和粗细
color_line = (0, 255, 0)  
thickness_line = 5
zoom_Z = 1

criterion = 10

# 设置点的颜色和大小
color_points = (0, 0, 255)  
radius_points = 10
# -1 表示实心圆
thickness_points = -1  

if_save = True
img_cv2_save_path = 'model/cv2_line_points.png'
if_show = True
if_show = False

new_valid_points, valid_x_y_z_edges, cv2_image, T_L = draw_save_cv2(data_coordinates, 
                                                                    coordinates_x_y_z,
                                                                    points_ori,
                                                                    valid_edges,
                                                                    zoomWH, 
                                                                    color_line, 
                                                                    thickness_line, 
                                                                    zoom_Z, 
                                                                    criterion,
                                                                    if_save,
                                                                    if_show,
                                                                    img_cv2_save_path,
                                                                    output_file_path,
                                                                    radius_points,
                                                                    color_points,
                                                                    thickness_points,
                                                                    start_time)

[
    para_draw_lines_CV2_T,
    para_draw_points_CV2_T,
    para_save_CV2_T
] = T_L

elapsed_time = showTime(start_time, '绘制CV2图与获取valid_x_y_z_edges-时间')
para_valid_x_y_z_edges_T = elapsed_time

# =====================================================================================
# 保存obj模型

# coordinates_x_y_z[:, 2] *= 3
coordinates_x_y_z_rotate = rotate_point_cloud(coordinates_x_y_z)

remove_time1 = time.time()

# print(f'{coordinates_x_y_z.shape[0] = }')
para_cloudPoints_N = coordinates_x_y_z.shape[0]

# ==================================================
# 测试不删除冗余的3D模型生成
obj_path = './model/model_has_redundant.obj'

if_show = False
mesh = save_show_obj(obj_path, coordinates_x_y_z_rotate, valid_simplex, if_show)

para_valid_simplex_N = len(valid_simplex)

elapsed_time = showTime(start_time, '不删除冗余-生成3D模型时间')
para_gen_model_has_redundant_model_T = elapsed_time


# 获取3D模型的正视图
mesh_cv2_img_has_redundant = render_front_view_to_cv2(mesh)
elapsed_time = showTime(start_time, '不删除冗余-获取3D模型的正视图的时间')
para_gen_model_has_redundant_fronView_T = elapsed_time

remove_time2 = time.time()

# ==================================================
# 测试删除冗余的3D模型生成
new_valid_points = list_to_np_array(new_valid_points)
valid_simplex_faces = get_valid_simplex_in_xyz(valid_simplex, coordinates_x_y_z, valid_x_y_z_edges)
coordinates_x_y_z = rotate_point_cloud(coordinates_x_y_z)

para_valid_simplex_faces_N = len(valid_simplex_faces)


obj_path = './model/model_remove_redundant.obj'

# Delete redundant mesh
if_show = False
mesh = save_show_obj(obj_path, coordinates_x_y_z, valid_simplex_faces, if_show)

elapsed_time = showTime_remove(start_time, remove_time1, remove_time2, '删除冗余-生成3D模型时间')
para_gen_model_remove_redundant_model_T = elapsed_time


# 获取3D模型的正视图
mesh_cv2_img_remove_redundant = render_front_view_to_cv2(mesh)
elapsed_time = showTime_remove(start_time, remove_time1, remove_time2,  '删除冗余-获取3D模型的正视图的时间')
para_gen_model_remove_redundant_fronView_T = elapsed_time

# ==================================================

elapsed_time = showTime(start_time, '生成3D模型-全部执行完成时间')
para_gen_model_all_T = elapsed_time

# =====================================================================================
# 拼接结果图片

# interPN: {para_interleave_delete_points_N:.2f}, \
# interT: {para_interleave_delete_T:.2f}, \

# text = f'ID: {para_imge_ID}, \
# zoomN: {para_zoomN}, \
# npyT1: {para_read_data_coordinates_T:.2f}, \
# npyT2: {para_read_data_segs_T:.2f}, \
# prePN: {para_pre_points_N}, \
# numbaT: {para_Numba_CUDA_T:.2f}, \
# pltT: {para_show_plt_fig_T:.2f}, \
# linesCVT: {para_draw_lines_CV2_T:.2f}, \
# pointsCVT: {para_draw_points_CV2_T:.2f}, \n \
# saveCVT: {para_save_CV2_T:.2f}, \
# VedgesT: {para_valid_x_y_z_edges_T:.2f}, \
# cloudPN: {para_cloudPoints_N}, \
# vSimplexN: {para_valid_simplex_N}, \
# yesReduMT: {para_gen_model_has_redundant_model_T:.2f}, \
# yesReduFT: {para_gen_model_has_redundant_fronView_T:.2f}, \
# reSimpleN: {para_valid_simplex_faces_N}, \
# noReduMT: {para_gen_model_remove_redundant_model_T:.2f}, \
# noReduFT: {para_gen_model_remove_redundant_fronView_T:.2f}, \
# genAllMT: {para_gen_model_all_T:.2f} \
# ', 


text = f'ID: {para_imge_ID}, \
zoomN: {para_zoomN}, \
numbaT: {para_Numba_CUDA_T:.2f}, \
pltT: {para_show_plt_fig_T:.2f}, \
saveCVT: {para_save_CV2_T:.2f}, \
vSimplexN: {para_valid_simplex_N}, \
reSimpleN: {para_valid_simplex_faces_N}, \
cloudPN: {para_cloudPoints_N}, \
yesReduMT: {para_gen_model_has_redundant_model_T:.2f}, \
yesReduFT: {para_gen_model_has_redundant_fronView_T:.2f}, \
noReduMT: {para_gen_model_remove_redundant_model_T:.2f}, \
noReduFT: {para_gen_model_remove_redundant_fronView_T:.2f}, \
genAllMT: {para_gen_model_all_T:.2f} \
', 

text = str(text)[2:-3] 

show_combined_cv2 = False
save_combined_cv2 = True
combined_img_path = './model/combined_image.jpg'

combined_cv2_imgs(combined_img_path, source_img, plt_fig, cv2_image, data_depth, mesh_cv2_img_has_redundant, mesh_cv2_img_remove_redundant, show_combined_cv2, save_combined_cv2, text)

showTime(start_time, '结果图片拼接执行->完成时间')

print(f'{text = }')






