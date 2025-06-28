import numpy as np
import os
import cv2

script_path = os.path.abspath(__file__)
print("脚本文件路径是:", script_path)
script_directory = os.path.dirname(script_path)
print("脚本所在目录是:", script_directory)

# 指定 .npy 文件的路径
# file_path = 'Outputs/reel1/sapiens_1b/000000_seg.npy'
# file_path = 'Outputs/reel1/sapiens_1b/000000.npy'
# file_path = 'Outputs/reel1/sapiens_2b/000001.npy'

# file_path = 'Outputs/reel1/sapiens_2b/depth/000000.npy'

file_path = 'Outputs/reel1/sapiens_2b/depth_test/processed/000099.npy'
# file_path = 'Outputs/reel1/sapiens_2b/depth_test/processed0/000001.npy'

file_path = os.path.join(script_directory, file_path)
print(f'{file_path = }')

# 使用 np.load() 函数读取文件
data = np.load(file_path)

# 打印数据
# print(f'{data = }')
print(f'{data.shape = }')
print(f'{data[0] = }')
print(f'{np.max(data) = }')
print(f'{np.min(data) = }')
# print(f'{np.where(data == np.max(data)) = }')
# print(f'{np.where(data == np.min(data)) = }')
# print(f'{len(np.where(data == np.max(data))[0]) = }')
# print(f'{len(np.where(data == np.min(data))[0]) = }')

imag_dir = 'Outputs/ReadNpy'
imag_name = 'readNpy.jpg'
imag_name_gray = 'readNpy_gray.jpg'
output_file_path = os.path.join(script_directory, imag_dir, imag_name)
output_file_path_gray = os.path.join(script_directory, imag_dir, imag_name_gray)

data_grey = np.dot(data[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

condition = data_grey > 100
# 使用 np.where 找到满足条件的元素的索引
x_indices, y_indices = np.where(condition)
coordinates = list(zip(x_indices, y_indices))
# print(f'{coordinates = }')

cv2.imwrite(output_file_path, data)
cv2.imwrite(output_file_path_gray, data_grey)


window_name = "Image Display"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

zoom = 5
cv2.resizeWindow(window_name, int(data.shape[1]/zoom), int(data.shape[0]/zoom))

# data[data == 100] = 0

cv2.imshow(window_name, data)

cv2.waitKey(0)
cv2.destroyAllWindows()

# print(f'{np.where(data == np.min(data)) = }')
# print(f'{np.where(data == np.max(data)) = }')