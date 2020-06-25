from __future__ import print_function
import sys
import h5py
import numpy as np
import cv2
import torch
import torch.utils.data as data
# from utils import np_skew_symmetric


def collate_fn(batch):
    batch_size = len(batch)
    numkps = np.array([sample['xs'].shape[1] for sample in batch])
    cur_num_kp = int(numkps.min())

    data = {}
    data['K1s'], data['K2s'], data['Rs'], \
        data['ts'], data['xs'], data['ys'], data['T1s'], data['T2s'], data['sides']  = [], [], [], [], [], [], [], [], []

    for sample in batch:
        data['K1s'].append(sample['K1'])
        data['K2s'].append(sample['K2'])
        data['T1s'].append(sample['T1'])
        data['T2s'].append(sample['T2'])
        data['Rs'].append(sample['R'])
        data['ts'].append(sample['t'])
        if sample['xs'].shape[1] > cur_num_kp:
            sub_idx = np.random.choice(sample['xs'].shape[1], cur_num_kp)
            data['xs'].append(sample['xs'][:,sub_idx,:])
            data['ys'].append(sample['ys'][sub_idx,:])
            if len(sample['side']) != 0:
                data['sides'].append(sample['side'][sub_idx,:])
        else:
            data['xs'].append(sample['xs'])
            data['ys'].append(sample['ys'])
            if len(sample['side']) != 0:
                data['sides'].append(sample['side'])


    # for key in ['K1s', 'K2s', 'Rs', 'ts', 'xs', 'ys', 'T1s', 'T2s','virtPts']:
    for key in ['K1s', 'K2s', 'Rs', 'ts', 'xs', 'ys', 'T1s', 'T2s']:
        data[key] = torch.from_numpy(np.stack(data[key])).float()
    if data['sides'] != []:
        data['sides'] = torch.from_numpy(np.stack(data['sides'])).float()
    return data



class CorrespondencesDataset(data.Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.data = None

   
    
    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.filename,'r')

        xs = np.asarray(self.data['xs'][str(index)])
        ys = np.asarray(self.data['ys'][str(index)])
        R = np.asarray(self.data['Rs'][str(index)])
        t = np.asarray(self.data['ts'][str(index)])
        side = []
        
        side.append(np.asarray(self.data['ratios'][str(index)]).reshape(-1,1)) 
        side.append(np.asarray(self.data['mutuals'][str(index)]).reshape(-1,1))
        side = np.concatenate(side,axis=-1)


        
        cx1 = np.asarray(self.data['cx1s'][str(index)])
        cy1 = np.asarray(self.data['cy1s'][str(index)])
        cx2 = np.asarray(self.data['cx2s'][str(index)])
        cy2 = np.asarray(self.data['cy2s'][str(index)])
        f1 = np.asarray(self.data['f1s'][str(index)])
        f2 = np.asarray(self.data['f2s'][str(index)])
        f1 = f1[0] if f1.ndim == 2 else f1
        f2 = f2[0] if f2.ndim == 2 else f2
        K1 = np.asarray([
            [f1[0], 0, cx1[0]],
            [0, f1[1], cy1[0]],
            [0, 0, 1]
            ])
        K2 = np.asarray([
            [f2[0], 0, cx2[0]],
            [0, f2[1], cy2[0]],
            [0, 0, 1]
            ])
       
        T1, T2 = np.zeros(1), np.zeros(1)


        return {'K1':K1, 'K2':K2, 'R':R, 't':t, \
        'xs':xs, 'ys':ys, 'T1':T1, 'T2':T2, 'side':side}
        
    def reset(self):
        if self.data is not None:
            self.data.close()
        self.data = None

    def __len__(self):
        if self.data is None:
            self.data = h5py.File(self.filename,'r')
            _len = len(self.data['xs'])
            self.data.close()
            self.data = None
        else:
            _len = len(self.data['xs'])
        return _len

    def __del__(self):
        if self.data is not None:
            self.data.close()

