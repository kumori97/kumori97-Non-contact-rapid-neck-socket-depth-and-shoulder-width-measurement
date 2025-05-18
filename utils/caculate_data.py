import torch
import os
from os.path import join, split, exists
import pickle as pkl
import numpy as np
from glob import glob
import trimesh
import codecs
# from kaolin.rep import TriangleMesh as tm
from psbody.mesh import Mesh

# Import the multiprocessing module
from multiprocessing import Pool

# Number of points to sample from the scan
NUM_POINTS = 15000


def remove_points_below_random_cutoff(path,input_scan,back_closest_points, roi, part_labels):
    # 1. 获取脚底最低点（Y轴方向）
    min_y_foot = np.min(input_scan.v[:, 1])

    # 2. 获取 roi 区域的最低点
    min_y_roi = np.min(roi[:, 1])

    # 3. 计算 H（roi 区域最低点到脚底最低点的高度）
    H = min_y_roi - min_y_foot

    # 4. 随机生成一个 0~0.6 之间的值
    random_ratio = np.random.uniform(0, 0.8)

    # 5. 根据随机值设置切割平面高度
    cutoff_height = min_y_foot + random_ratio * H

    # 6. 找出高于该高度的点索引
    points_above_cutoff = input_scan.v[:, 1] >= cutoff_height

    # 7. 过滤掉低于该高度的点云和部件标签
    filtered_back_closest_points = back_closest_points.v[points_above_cutoff]
    filtered_scan_v = input_scan.v[points_above_cutoff]
    filtered_part_labels = part_labels[points_above_cutoff]

    # 8. 更新扫描数据和部件标签
    input_scan.v = filtered_scan_v
    filtered_mesh = Mesh(v=filtered_scan_v, f=[])
    filtered_back_closest_points = Mesh(v=filtered_back_closest_points, f=[])

    # 9. 保存过滤后的点云和标签数据
    filtered_mesh.set_vertex_colors_from_weights(filtered_part_labels.flatten())
    filtered_mesh.write_ply(join(path, 'back_views', 'filtered_0_back_color.ply'))
    filtered_back_closest_points.set_vertex_colors_from_weights(filtered_part_labels.flatten())
    filtered_back_closest_points.write_ply(join(path, 'back_views', 'filtered_back_inner_points.ply'))
    np.savetxt(join(path, 'back_views', 'filtered_part_labels.txt'), filtered_part_labels.astype('float32'))


    #10. 随机采样10000个点(测试)
    num_samples = 10000
    rand_indices = np.random.choice(len(filtered_mesh.v), num_samples)

    scan_sampled =  Mesh(v=np.asarray(filtered_mesh.v)[rand_indices], f=[])
    back_inner_sampled = Mesh(v=np.asarray(filtered_back_closest_points.v)[rand_indices], f=[])
    labels_sampled = filtered_part_labels[rand_indices]

    scan_sampled.set_vertex_colors_from_weights(labels_sampled.flatten())
    scan_sampled.write_ply(join(path, 'back_views', 'filtered_0_back_color_sampled.ply'))
    back_inner_sampled.set_vertex_colors_from_weights(labels_sampled.flatten())
    back_inner_sampled.write_ply(join(path, 'back_views', 'filtered_back_inner_points_sampled.ply'))
    np.savetxt(join(path, 'back_views', 'filtered_part_labels_sampled.txt'), labels_sampled.astype('float32'))

    return filtered_mesh,filtered_back_closest_points, filtered_part_labels


def process_file(path):
    print(path)
    name = split(path)[-1]
    # Load smpl part labels
    with open('../assets/smpl_parts_four.pkl', 'rb') as f:         ##############################################################
        dat = pkl.load(f, encoding='latin-1')
    smpl_parts = np.zeros((6890, 1))
    for n, k in enumerate(dat):
        smpl_parts[dat[k]] = n

    input_smpl = Mesh(filename=join(path, name + '_naked.obj'))
    R = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])[:3,:3]
    input_smpl.v = input_smpl.v.dot(R)
    input_scan = Mesh(filename=join(path,'back_views','0_back.ply'))
    ind, _ = input_smpl.closest_vertices(input_scan.v)
    part_labels = smpl_parts[np.array(ind)]

    #################查找back和neck的索引
    roi = input_scan.v[(part_labels.flatten()==0) | (part_labels.flatten()==1)|(part_labels.flatten()==2)|(part_labels.flatten()==3)]
    closest_face, closest_points = input_smpl.closest_faces_and_points(roi)
    roi_closest_points = Mesh(v=closest_points,f=[])
    roi_closest_points.write_ply(join(path, 'back_views', 'roi_inner_points.ply'))
    ##################

    ##########计算背部点云的内部点#############
    closest_face, closest_points = input_smpl.closest_faces_and_points(input_scan.v)
    back_closest_points = Mesh(v=closest_points,f=[])
    back_closest_points.set_vertex_colors_from_weights(part_labels.flatten())
    back_closest_points.write_ply(join(path, 'back_views', 'back_inner_points.ply'))
    ############################

    # input_scan.set_vertex_colors(part_labels)
    input_scan.set_vertex_colors_from_weights(part_labels.flatten())
    input_scan.write_ply(join(path, 'back_views','0_back_color.ply'))
    input_smpl.write_ply(join(path, 'back_views','input_smpl_back.ply'))
    np.savetxt(join(path,'back_views','part_labels.txt'), part_labels.astype('float32'))

    #删除部分点
    filtered_mesh,filtered_back_closest_points, filtered_part_labels = remove_points_below_random_cutoff(path,input_scan,back_closest_points, roi, part_labels)




if __name__ == "__main__":
    #用python3.7跑多进程会卡住
    npz_files = glob('../assets/CAPE_Dataset_Sampled_10000/**/*_naked.obj', recursive=True)

    npz_files = [split(file)[0] for file in npz_files]

    # Create a pool of worker processes
    pool = Pool(processes=18)
    # Use the pool to process the files in parallel
    pool.map(process_file, npz_files)
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    #
    # for item in npz_files:
    #     process_file(item)
    #
