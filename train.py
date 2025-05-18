"""
Code to warm start the correspondence prediction network with small amount of supervised data.
Author: Bharat
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
"""

import os
from os.path import split, join, exists
import torch
from models.trainer import Trainer_Inner_back_net
############
from data_loader import MyDataLoader_back
from models.Scan2Back_net2 import Scan2Back_net



def main_back(mode,  optimizer, batch_size, epochs,  exp_name ='part_specific_net',
         split_file=None):
    if split_file is None:
        split_file = 'assets/dataset_split.pkl'

    # corr_net = PointNet2Part(in_features=0, num_parts=5, num_classes=3)
    corr_net =Scan2Back_net(c=3, k=5)
    exp_name = exp_name

    if mode == 'train':
        dataset = MyDataLoader_back('train', batch_size, num_workers=0,split_file=split_file).get_loader()
        trainer = Trainer_Inner_back_net(corr_net, torch.device("cuda"), dataset, None, exp_name, optimizer=optimizer)
        trainer.train_model(epochs)
    elif mode == 'val':
        dataset = MyDataLoader_back('val', batch_size, num_workers=8,
                                 split_file=split_file).get_loader(shuffle=False)
        trainer = Trainer_Inner_back_net(corr_net, torch.device("cuda"), None, dataset, exp_name,
                                         optimizer=optimizer)
        trainer.eval_model(mode)
    else:
        print('Invalid mode. should be either train, val or eval.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('-exp_name',default='Inner_back_cape_2', type=str)
    parser.add_argument('-batch_size', default=12, type=int)
    parser.add_argument('-optimizer', default='Adam', type=str)
    parser.add_argument('-epochs', default=1000, type=int)
    parser.add_argument('-augment', default=False, action='store_true')
    # Train network for dressed or undressed scans
    parser.add_argument('-split_file', type=str, default='assets/CAPE_Dataset_Sampled_10000.pkl')
    # Validation specific arguments
    parser.add_argument('-mode', default='train', choices=['train', 'val', 'eval'])
    parser.add_argument('-save_name', default='', type=str)
    parser.add_argument('-num_saves', default=5, type=int)

    args = parser.parse_args()

    if args.mode == 'val':
        # assert len(args.save_name) > 0
        main_back('val', args.optimizer, args.batch_size, args.epochs,
                 exp_name=args.exp_name, split_file=args.split_file)
    elif args.mode == 'train':
        main_back('train', args.optimizer, args.batch_size, args.epochs, exp_name=args.exp_name,
                 split_file=args.split_file)

