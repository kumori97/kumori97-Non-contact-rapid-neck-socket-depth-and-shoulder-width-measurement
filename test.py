
import numpy as np
import torch
import trimesh

import open3d as o3d
import pyransac3d as pyrsc
np.random.seed(100)

#导入和处理数据
back_scan = trimesh.load('0_back.ply')
back_scan = back_scan.vertices[np.random.randint(0, len(back_scan.vertices), 10000)]
pcd_back_scan = back_scan.copy()

# 标准化数据
center = np.mean(back_scan, axis=0)
back_scan -= center
scale_factor = 1.5 / (np.max(back_scan[:, 1]) - np.min(back_scan[:, 1]))
back_scan *= scale_factor

back_scan = torch.from_numpy(back_scan).unsqueeze(0).to(torch.float32).cuda()

# model初始化
from models.Scan2Back_net2 import Scan2Back_net
corr_net = Scan2Back_net(c=3, k=5).cuda()
checkpoint = torch.load('experiments/Inner_back_cape_2/checkpoints/checkpoint_epoch_999.tar')
corr_net.load_state_dict(checkpoint['model_state_dict'])
corr_net.eval()
#模型推理
out = corr_net(back_scan)

#获取背部点云的标签
part_labels = out['part_labels']
_, part_label = torch.max(part_labels.data, 1)
part_label = np.array(part_label.cpu(), dtype='int32')
part_label = part_label.reshape(-1)

#获取背部内部点ROI
back_inner_points = out['back_inner_points'].detach().permute(0, 2, 1).cpu().numpy()[0]/scale_factor + center
back_inner_points_roi = back_inner_points[part_label == 0]
temp = o3d.geometry.PointCloud()
temp.points = o3d.utility.Vector3dVector(back_inner_points_roi)
cl, ind = temp.remove_radius_outlier(nb_points=5, radius=0.01)
back_inner_points_roi =  np.asarray(cl.points)

#获取颈部内部点ROI
neck_inner_points_roi = back_inner_points[part_label == 1]
temp = o3d.geometry.PointCloud()
temp.points = o3d.utility.Vector3dVector(neck_inner_points_roi)
cl, ind = temp.remove_radius_outlier(nb_points=3, radius=0.01)
neck_inner_points_roi =  np.asarray(cl.points)

#获取左右肩部的ROI
left_deltoid_muscle_inner_points_roi = back_inner_points[part_label == 2]
right_deltoid_muscle_inner_points_roi = back_inner_points[part_label == 3]


#获取背部拟合平面
plano1 = pyrsc.Plane()
best_eq, best_inliers = plano1.fit(back_inner_points_roi, 0.01)

#计算颈窝深度
a, b, c, d = best_eq
# 计算每个点的深度
socket_depth_list = np.abs(a * neck_inner_points_roi[:, 0] + b * neck_inner_points_roi[:, 1]
                           + c * neck_inner_points_roi[:, 2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
socket_depth = np.max(socket_depth_list)
max_depth_index = np.argmax(socket_depth_list)
max_depth_point = neck_inner_points_roi[max_depth_index]
print('socket_depth: {}mm'.format(socket_depth*1000))


#########计算肩最大宽###########
back_inner_points_left_deltoid_muscle = back_inner_points[part_label == 2]
back_inner_points_right_deltoid_muscle = back_inner_points[part_label == 3]
#将肩部点投影到xz平面
back_inner_points_left_deltoid_muscle_xz = back_inner_points_left_deltoid_muscle[:,[0,2]]
back_inner_points_right_deltoid_muscle_xz = back_inner_points_right_deltoid_muscle[:,[0,2]]
# 计算两个点云之间的所有点对的距离
distances = np.sqrt(np.sum((back_inner_points_left_deltoid_muscle_xz [:, np.newaxis] - back_inner_points_right_deltoid_muscle_xz)**2, axis=2))
# 找到距离最大的点对
max_distance = np.max(distances)
max_distance_indices = np.unravel_index(np.argmax(distances), distances.shape)
# 打印结果
print("Maximum shoulder width: {}mm".format(max_distance*1000))

#线段的两个端点
aa = back_inner_points_left_deltoid_muscle[max_distance_indices[0].item()]
bb = back_inner_points_right_deltoid_muscle[max_distance_indices[1].item()]
aa[1] = (aa[1] + bb[1])/2
bb[1] = (aa[1] + bb[1])/2

#计算颈窝深与背部平面的交点
import math
# 点坐标
x0, y0, z0 = max_depth_point[0],max_depth_point[1],max_depth_point[2]
# 求法向量
n = (a, b, c)
# 计算点到平面的距离 t
t = (a*x0 + b*y0 + c*z0 + d) / math.sqrt(a**2 + b**2 + c**2)
# 计算交点坐标
x = x0 - a*t
y = y0 - b*t
z = z0 - c*t
# print(f'交点坐标: ({x:.3f},{y:.3f},{z:.3f})')
# print(a*x+b*y+c*z+d)
end_point = np.array([x,y,z])



########################################open3d显示##################################################
# 创建 PointCloud 对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(back_inner_points_roi)
plane = pcd.select_by_index(best_inliers).paint_uniform_color([0, 0, 0.5])
obb = plane.get_oriented_bounding_box()
obb2 = plane.get_axis_aligned_bounding_box()
obb.color = [0, 0, 1]
obb2.color = [0, 1, 0]
not_plane = pcd.select_by_index(best_inliers, invert=True)

# best_eq 是平面的参数，形式为 [a, b, c, d]，对应平面方程 ax + by + cz + d = 0
a, b, c, d = best_eq
# 创建 x, y 的网格
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
x, y = np.meshgrid(x, y)
z = (-d - a*x - b*y) / c
mesh = o3d.geometry.TriangleMesh()
# 使用 best_eq 定义的平面参数来创建一个网格
# 这里我们假设 x, y, z 是你之前计算出的网格坐标
for i in range(x.shape[0] - 1):
    for j in range(x.shape[1] - 1):
        # 添加每个小方格的两个三角形
        mesh.triangles.append([i * x.shape[1] + j, (i + 1) * x.shape[1] + j, i * x.shape[1] + j + 1])
        mesh.triangles.append([(i + 1) * x.shape[1] + j, (i + 1) * x.shape[1] + j + 1, i * x.shape[1] + j + 1])
# 添加顶点坐标
mesh.vertices = o3d.utility.Vector3dVector(np.vstack([x.flatten(), y.flatten(), z.flatten()]).T)

#绘制坐标轴
coordinate_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])


#导入整个背部内部点云
pcd_back_scan_inner = o3d.geometry.PointCloud()
pcd_back_scan_inner.points = o3d.utility.Vector3dVector(back_inner_points)

#导入整个背部外部点云
pcd_back_scan_outer = o3d.geometry.PointCloud()
pcd_back_scan_outer.points = o3d.utility.Vector3dVector(pcd_back_scan)
pcd_back_scan_outer.paint_uniform_color([0,0,0])
#导入颈部点云
pcd_neck_inner = o3d.geometry.PointCloud()
pcd_neck_inner.points = o3d.utility.Vector3dVector(neck_inner_points_roi)
pcd_neck_inner.paint_uniform_color([0,0.5,1])
#导入左右肩点云
left_deltoid_muscle_inner = o3d.geometry.PointCloud()
left_deltoid_muscle_inner.points = o3d.utility.Vector3dVector(left_deltoid_muscle_inner_points_roi)
left_deltoid_muscle_inner.paint_uniform_color([0.5,1,0.5])
right_deltoid_muscle_inner = o3d.geometry.PointCloud()
right_deltoid_muscle_inner.points = o3d.utility.Vector3dVector(right_deltoid_muscle_inner_points_roi)
right_deltoid_muscle_inner.paint_uniform_color([1,0.58,0])

# 创建 LineSet 对象(颈窝线)
lineset = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([max_depth_point, end_point]),
    lines=o3d.utility.Vector2iVector([[0, 1]])
)
# 创建 LineSet 对象(最大肩宽线)
lineset2 = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([aa, bb]),
    lines=o3d.utility.Vector2iVector([[0, 1]])
)

# o3d.visualization.draw_geometries([plane,pcd_neck_inner,coordinate_axis,mesh,lineset,pcd_back_scan_outer,
#                                    left_deltoid_muscle_inner,right_deltoid_muscle_inner,lineset2])
o3d.visualization.draw_geometries([plane,pcd_neck_inner,mesh,lineset,
                                   left_deltoid_muscle_inner,right_deltoid_muscle_inner,lineset2])