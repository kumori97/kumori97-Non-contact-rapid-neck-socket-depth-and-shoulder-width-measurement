import torch
import os
from os.path import join, split, exists
import pickle as pkl
import numpy as np
import codecs
from glob import glob

import trimesh
from torch.utils.data import Dataset, DataLoader
# Number of points to sample from the scan

class MyDataLoader_back(Dataset):
    def __init__(self, mode, batch_sz,
                 split_file='../assets/dataset_split.pkl', num_workers=12,
                 ):
        self.mode = mode
        with open(split_file, "rb") as f:
            self.split = pkl.load(f)

        self.data = self.split[mode]
        self.batch_size = batch_sz
        self.num_workers = num_workers


    def __len__(self):
        return len(self.data)

    def get_loader(self, shuffle=True):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    @staticmethod
    def worker_init_fn(worker_id):
        """
        Worker init function to ensure true randomness.
        """
        base_seed = int(codecs.encode(os.urandom(4), 'hex'), 16)
        np.random.seed(base_seed + worker_id)

    def __getitem__(self, idx):
        path = self.data[idx]
        # print(path)
        scan = np.asarray(trimesh.load(join(path,'back_views','filtered_0_back_color.ply')).vertices)
        part_labels = np.loadtxt(join(path, 'back_views','filtered_part_labels.txt'))
        back_inner_points = np.asarray(trimesh.load(join(path,'back_views','filtered_back_inner_points.ply')).vertices)

        #标准化数据
        center = np.mean(scan, axis=0)
        scan -= center
        scale_factor = 1.5 / (np.max(scan[:, 1]) - np.min(scan[:, 1]))
        scan *= scale_factor
        back_inner_points -= center
        back_inner_points *= scale_factor


        num_samples = 10000
        rand_indices = np.random.choice(len(scan), num_samples)


        scan_sampled = scan[rand_indices]
        labels_sampled = part_labels[rand_indices]
        back_inner_sampled = back_inner_points[rand_indices]

        return {'scan': scan_sampled.astype('float32'),
                'part_labels': labels_sampled.astype('float32'),
                'back_inner': back_inner_sampled.astype('float32'),
                'path': path
                }
















if __name__ =="__main__":
    train_dataset = MyDataLoader_back('train',3, split_file='./assets/dataset_split.pkl',num_workers=0)
    train_dataloader = train_dataset.get_loader()
    for dic in train_dataloader:
        print(dic['path'])