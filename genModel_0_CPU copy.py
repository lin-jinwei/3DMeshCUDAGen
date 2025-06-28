import os
import numpy as np
from numba import cuda, float32
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



# ============================================================================
# ============================================================================
# Functions Definition

def assign_clusters_cpu(points, centroids):
    # 计算每个点到所有聚类中心的距离，并分配到最近的中心
    labels = np.argmin(np.sum((points[:, np.newaxis, :] - centroids) ** 2, axis=2), axis=1)
    return labels

def update_centroids_cpu(points, centroids, labels):
    # 更新聚类中心
    for i in range(len(centroids)):
        mask = labels == i
        if np.any(mask):  # 检查是否至少有一个点属于该簇
            centroids[i] = np.mean(points[mask], axis=0)

def kmeans_numba_cpu(points, num_clusters, max_iter=10):
    # 初始化
    np.random.seed(0)  # 设置随机种子以保证结果可重复
    centroids_indices = np.random.choice(points.shape[0], num_clusters, replace=False)
    centroids = points[centroids_indices].copy()

    for _ in range(max_iter):
        labels = assign_clusters_cpu(points, centroids)
        old_centroids = centroids.copy()
        update_centroids_cpu(points, centroids, labels)
        
        # 可选: 检查收敛条件
        if np.allclose(old_centroids, centroids):
            break

    return centroids


def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def find_pairs(points):
    num_points = len(points)
    used = [False] * num_points
    pairs = []
    
    # 对所有点进行配对
    for i in range(num_points):
        if not used[i]:
            nearest_point = None
            min_distance = float('inf')
            for j in range(num_points):
                if i != j and not used[j]:
                    distance = calculate_distance(points[i], points[j])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_point = j
            # 标记这对点为已使用
            used[i] = True
            used[nearest_point] = True
            pairs.append((points[i], points[nearest_point]))
    
    return pairs


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
        
        
def get_npy_data(file_path):
    file_path = os.path.join(script_directory, file_path)
    data = np.load(file_path)
    return data


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
    

def get_coordinates(data):
    data_grey = np.dot(data[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    showTime(start_time, '转换灰度图时间')

    condition = data_grey > 0
    x_indices, y_indices = np.where(condition)
    coordinates_x_y = list(zip(x_indices, y_indices))

    showTime(start_time, '获取 coordinates_x_y 时间')

    coordinates_x_y_z = []
    for x_y in coordinates_x_y:
        z = data_grey[x_y]
        coordinates_x_y_z.append([x_y[0], x_y[1], z])

    showTime(start_time, '获取 coordinates_x_y_z 时间')
    
    return coordinates_x_y, coordinates_x_y_z


def get_coordinates_segs(data_coordinates, data_segs):
    data_grey = np.dot(data_coordinates[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    showTime(start_time, '转换灰度图时间')

    condition = data_grey > 0
    x_indices, y_indices = np.where(condition)
    coordinates_x_y = list(zip(x_indices, y_indices))

    showTime(start_time, '获取 coordinates_x_y 时间')

    coordinates_x_y_z = []
    Dict_x_y_z_segs = {}
    for x_y in coordinates_x_y:
        z = data_grey[x_y]
        seg = data_segs[x_y]
        coordinates_x_y_z.append([x_y[0], x_y[1], z])
        Dict_x_y_z_segs[(x_y[0], x_y[1])] = [z, seg]

    showTime(start_time, '获取 coordinates_x_y_z + Dict_x_y_z_segs 时间')
    return coordinates_x_y, coordinates_x_y_z, Dict_x_y_z_segs


def interleave_delete(coordinates_x_y_z, gap):
    coordinates_x_y_z = np.array(coordinates_x_y_z)
    rows_to_delete = np.arange(0, coordinates_x_y_z.shape[0], gap)
    coordinates_x_y_z = np.delete(coordinates_x_y_z, rows_to_delete, axis=0)  # 删除行
    coordinates_x_y_z = coordinates_x_y_z.tolist()
    return coordinates_x_y_z  


def centroids_cluster(coordinates_x_y_z, zoomN):
    num_clusters = int(len(coordinates_x_y_z) / zoomN)
    coordinates_x_y_z = np.array(coordinates_x_y_z, dtype=np.float32)
    centroids = kmeans_numba_cpu(coordinates_x_y_z, num_clusters)
    return centroids
     

def get_points_pointsOri(points):
    points2 = np.array([(y, -x) for x, y in points])
    points_ori = np.array([(x, -y) for x, y in points2])
    return points2, points_ori


def distance_matrix_Delaunay(points):
    # 使用scipy.spatial.Delaunay进行Delaunay三角剖分
    # 创建 Delaunay 三角剖分对象
    tri = Delaunay(points)
    # 计算所有点之间的距离矩阵
    distance_matrix = squareform(pdist(points))
    return tri, distance_matrix


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


def divide_segment(x1, y1, x2, y2, n):
    if n <= 0:
        raise ValueError("n 必须是一个正整数")
    # 计算每个小线段的增量
    dx = (x2 - x1) / n
    dy = (y2 - y1) / n
    # 创建分割点列表，包含起始点和结束点
    points = []
    for i in range(n + 1):
        new_x = x1 + i * dx
        new_y = y1 + i * dy
        points.append((new_x, new_y))
    return points

    
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
                  ):
    
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
    # return  valid_x_y_z_edges
        
    
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


def has_duplicates(list_A):
    seen = set()  # 用于存储已经遇到的元素
    for element in list_A:
        try:
            # 将 [a, b, c] 转换为元组以确保其可哈希
            element_tuple = tuple(element)
            if element_tuple in seen:
                return True
            seen.add(element_tuple)
        except TypeError:
            print(f"Element {element} cannot be converted to a tuple.")
            continue
    return False


def swap_x_y(input_list):
    if not isinstance(input_list, list) or len(input_list) != 3:
        raise ValueError("Input must be a list of three elements.")
    # 解构赋值方式交换 x1 和 y1 的位置
    x1, y1, z1 = input_list
    swapped_list = [y1, x1, z1]
    return swapped_list


def list_to_np_array(input_list, dtype=None):
    return np.array(input_list, dtype=dtype)


def get_valid_simplex_in_xyz(valid_simplex, coordinates_x_y_z, valid_x_y_z_edges):
    valid_simplex_faces = []
    for a_b_c in valid_simplex:
        pa = coordinates_x_y_z[a_b_c[0]]
        pb = coordinates_x_y_z[a_b_c[1]]
        pc = coordinates_x_y_z[a_b_c[2]]
        
        pa_x_y = pa[0:1]
        pb_x_y = pb[0:1]
        pc_x_y = pc[0:1]
        
        flag_connected_a_b = False
        flag_connected_a_c = False
        flag_connected_b_c = False
    
        # print()
        # print(f'{pa = }')
        # print(f'{pb = }')
        # print(f'{pc = }')
        
        for x_y_z_edges in valid_x_y_z_edges:
            x_y_edges_0 = x_y_z_edges[0][0:1]
            x_y_edges_1 = x_y_z_edges[1][0:1]
            L_start_end_points = [x_y_edges_0, x_y_edges_1]
            
            if pa_x_y in L_start_end_points and pb_x_y in L_start_end_points:
                flag_connected_a_b = True
                # print(f'{pa_x_y} + {pb_x_y} in, flag_connected_a_b = {flag_connected_a_b}')
            
            if pa_x_y in L_start_end_points and pc_x_y in L_start_end_points:
                flag_connected_a_c = True
                # print(f'{pa_x_y} + {pc_x_y} in, flag_connected_a_b = {flag_connected_a_c}')

            if pb_x_y in L_start_end_points and pc_x_y in L_start_end_points:
                flag_connected_b_c = True
                # print(f'{pb_x_y} + {pc_x_y} in, flag_connected_a_b = {flag_connected_b_c}')
        
        if (flag_connected_a_b and flag_connected_a_c and flag_connected_b_c):
            # print(f'all True: {a_b_c = }')
            valid_simplex_faces.append(a_b_c)
    return valid_simplex_faces



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

zoomN = 10000
# zoomN = 5000
# zoomN = 2500
# zoomN = 2000
# zoomN = 1500

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
print(f'{data_coordinates.shape = }')

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
# # 栅格删除法
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
                                                                    )

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
# mesh = save_show_obj(obj_path, coordinates_x_y_z_rotate, valid_simplex, if_show)
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

text = f'ID: {para_imge_ID}, \
zoomN: {para_zoomN}, \
npyT1: {para_read_data_coordinates_T:.2f}, \
npyT2: {para_read_data_segs_T:.2f}, \
prePN: {para_pre_points_N}, \
numbaT: {para_Numba_CUDA_T:.2f}, \
pltT: {para_show_plt_fig_T:.2f}, \
linesCVT: {para_draw_lines_CV2_T:.2f}, \
pointsCVT: {para_draw_points_CV2_T:.2f}, \
saveCVT: {para_save_CV2_T:.2f}, \
VedgesT: {para_valid_x_y_z_edges_T:.2f}, \
cloudPN: {para_cloudPoints_N}, \
vSimplexN: {para_valid_simplex_N}, \
yesReduMT: {para_gen_model_has_redundant_model_T:.2f}, \
yesReduFT: {para_gen_model_has_redundant_fronView_T:.2f}, \
reSimpleN: {para_valid_simplex_faces_N}, \
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








