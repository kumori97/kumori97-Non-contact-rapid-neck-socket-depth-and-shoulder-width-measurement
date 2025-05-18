import numpy as np
import trimesh

import os
from os.path import join, split, exists


def get_3DSV(mesh,rot_y=0):
    from opendr.camera import ProjectPoints
    from opendr.renderer import DepthRenderer

    WIDTH, HEIGHT = 1000, 1000

    camera = ProjectPoints(v=mesh.vertices, f=np.array([WIDTH, WIDTH]), c=np.array([WIDTH, HEIGHT]) / 2.,
                           t=np.array([0, 0, 3]), rt=np.array([0,np.pi+(np.pi * rot_y / 180), 0]), k=np.zeros(5))
    frustum = {'near': 1., 'far': 10., 'width': WIDTH, 'height': HEIGHT}
    rn = DepthRenderer(camera=camera, frustum=frustum, f=mesh.faces, overdraw=False)

    points3d = camera.unproject_depth_image(rn.r)

    distances = np.linalg.norm(points3d, axis=2)

    keep_mask = distances <= 2.5

    # print(points3d[:, :, 2] > np.min(points3d[:, :, 2]) + 0.01)
    # points3d = points3d[points3d[:, :, 2] > np.min(points3d[:, :, 2]) + 0.01]
    points3d = points3d[keep_mask]
    num = np.random.rand()
    if num <0.33333:
        points3d_noise =  points3d
    elif num>=0.33333 and num <0.66666:
        points3d_noise =  points3d + 0.001 * np.random.randn(points3d.shape[0], 3)
    else:
        points3d_noise = points3d + 0.0015 * np.random.randn(points3d.shape[0], 3)
    print('sampled {} points.'.format(points3d.shape[0]))
    return points3d,points3d_noise


def main(path):
    random_rotationy = np.random.randint(-15, 15)
    mesh = trimesh.load(path)
    R = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
    mesh.apply_transform(R)
    points3d,points3d_noise = get_3DSV(mesh,rot_y = random_rotationy)

    mm = trimesh.Trimesh(vertices=points3d,faces=[])
    mm.export('0_back.ply')
    mm = trimesh.Trimesh(vertices=points3d_noise,faces=[])
    mm.export('0_back_noise.ply')


if __name__ == '__main__':
    main('0106.obj')
