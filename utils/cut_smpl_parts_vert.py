import open3d as o3d
import numpy as np
import os
import pickle as pkl


template_pcd = o3d.io.read_point_cloud("../assets/template_female.ply")
back_part_pcd = o3d.io.read_point_cloud("../assets/back.ply")
neck_part_pcd = o3d.io.read_point_cloud("../assets/neck.ply")
left_deltoid_muscle_pcd = o3d.io.read_point_cloud("../assets/left_deltoid_muscle.ply")
right_deltoid_muscle_pcd = o3d.io.read_point_cloud("../assets/right_deltoid_muscle.ply")
# 创建 k-d 树
kdtree = o3d.geometry.KDTreeFlann(template_pcd)

# 存储索引
back_indices = []
neck_indices = []
left_deltoid_muscle_indices = []
right_deltoid_muscle_indices = []

# 对于 back_part 中的每个点，找到在 template 中的最近点
for point in back_part_pcd.points:
    _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
    back_indices.append(idx[0])

for point in neck_part_pcd.points:
    _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
    neck_indices.append(idx[0])

for point in left_deltoid_muscle_pcd.points:
    _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
    left_deltoid_muscle_indices.append(idx[0])

for point in right_deltoid_muscle_pcd.points:
    _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
    right_deltoid_muscle_indices.append(idx[0])

# 现在，back_indices 包含了 back_part 中每个点在 template 中的索引
print(len(back_indices))
print(len(neck_indices))
print(len(left_deltoid_muscle_indices))
print(len(right_deltoid_muscle_indices))

all_indices = set(range(6890))
back_indices_set = set(back_indices)
neck_indices_set = set(neck_indices)
left_deltoid_muscle_indices_set = set(left_deltoid_muscle_indices)
right_deltoid_muscle_indices_set = set(right_deltoid_muscle_indices)
remaining_indices = all_indices - neck_indices_set - back_indices_set - left_deltoid_muscle_indices_set - right_deltoid_muscle_indices_set


back_indices = np.array(back_indices)
neck_indices = np.array(neck_indices)
left_deltoid_muscle_indices = np.array(left_deltoid_muscle_indices)
right_deltoid_muscle_indices = np.array(right_deltoid_muscle_indices)

other = np.array(list(remaining_indices))

parts = {'back': back_indices,'neck': neck_indices,'left_deltoid_muscle':left_deltoid_muscle_indices,
         'right_deltoid_muscle':right_deltoid_muscle_indices,'other': other}

col = np.zeros((6890,))
col[parts['back']] = 0
col[parts['neck']] = 1
col[parts['left_deltoid_muscle']] = 2
col[parts['right_deltoid_muscle']] = 3
col[parts['other']] = 4

import collections

# parts = collections.OrderedDict(sorted(parts.items()))

col = np.zeros((6890,))
for n, k in enumerate(parts):
    col[parts[k]] = n

pkl.dump(parts, open('../assets/smpl_parts_four.pkl', 'wb'))
