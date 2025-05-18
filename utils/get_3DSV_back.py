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
    # print random_rotationy

    mesh = trimesh.load(os.path.join(path,split(path)[-1]+'.obj'), process=False)

    R = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
    mesh.apply_transform(R)


    points3d,points3d_noise = get_3DSV(mesh,rot_y = random_rotationy)
    if exists(join(path,'back_views')) is False:
        os.mkdir(join(path,'back_views'))


    # np.savetxt(os.path.join(path, 'back_views','{}_back.txt'.format(i)), points3d,delimiter=';')
    # np.savetxt(os.path.join(path, 'back_views', '{}_back_noise.txt'.format(i)), points3d_noise, delimiter=';')


    mm = trimesh.Trimesh(vertices=points3d,faces=[])
    mm.export(os.path.join(path, 'back_views','0_back.ply'))
    mm = trimesh.Trimesh(vertices=points3d_noise,faces=[])
    mm.export(os.path.join(path, 'back_views','0_back_noise.ply'))


    # mm = trimesh.Trimesh(vertices=points3d,faces=[])
    # mm.export(os.path.join(path, 'back_{}.ply'.format(i)))
    # mm = trimesh.Trimesh(vertices=points3d_noise,faces=[])
    # mm.export(os.path.join(path, 'back_{}_noise.ply'.format(i)))





if __name__ == '__main__':
    from multiprocessing import Process, Pool
    # import glob
    # INPUT_PATH = '../assets/CAPE_Dataset_Sampled_10000/'
    # paths = np.sort(glob.glob(INPUT_PATH + '/*'))
    # np.random.seed(366666666)
    # for path_base in paths:
    #     path_base = np.sort(glob.glob(path_base + '/*'))
    #     for path in path_base:
    #         path = np.sort(glob.glob(path + '/*'))
    #         for num,mesh_path in enumerate(path):
    #             main(mesh_path,sample_num=1)
    #             print(mesh_path)


    from glob import glob
    npz_files = glob('../assets/CAPE_Dataset_Sampled_10000/**/*_naked.obj', recursive=True)
    npz_files = [split(file)[0] for file in npz_files]
    # Create a pool of worker processes
    pool = Pool(processes=6)
    # Use the pool to process the files in parallel
    pool.map(main, npz_files)
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()
