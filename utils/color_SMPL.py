import numpy as np
import pickle as pkl
from psbody.mesh import Mesh
import argparse
from os.path import split, join, exists

smpl_model = Mesh(filename='../assets/template_female.ply')
verts = np.array(smpl_model.v)  # 每个元素是输入点集中对应点在 garment_mesh.v 中的最近顶点的索引

part_labels = pkl.load(open('../assets/smpl_parts_four.pkl', 'rb'))
labels = np.zeros((6890,), dtype='int32')
for n, k in enumerate(part_labels):
    labels[part_labels[k]] = n

smpl_model.set_vertex_colors_from_weights(labels)
smpl_model.write_ply('seg_smple_test.ply')