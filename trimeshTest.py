import trimesh

# 示例点数据
vertices = [
    [0.123456, 0.0, 0.0],
    [1.123456, 0.0, 0.0],
    [0.123456, 1.0, 0.0],
]

# 示例面数据
# faces = [
#     [0, 1, 2],
#     [0, 2, 3],
#     # [1, 2, 3],
#     [0, 1, 3],
#     [3, 2, 1],
# ]

faces = [
    [0, 1, 2],
    # [0, 2, 3],
    # # [1, 2, 3],
    # [0, 1, 3],
    # [3, 2, 1],
]

# 创建一个Trimesh对象
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# 可视化模型
mesh.show()

# 保存模型为OBJ文件
mesh.export('model.obj')