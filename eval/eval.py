import argparse
import numpy as np
import os
import torch
import time
import cv2
from data import collate_fn, CorrespondencesDataset
from evaluation import eval_decompose, compute_metric


def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='eval essential matrix')
parser.add_argument('--data_te', type=str, default='../../data_dump/yfcc-sift-2000-test.hdf5',
  help='datasets path.')
parser.add_argument('--ratio_test_th', type=float, default=0.8,
  help='ratio test threshold')
parser.add_argument("--ransac_type", type=int, default=2, help=""
    "0:opencv ransac, 1:magsac, 2:gcransac")
parser.add_argument("--ransac_th", type=float, default=0.00025, help=""
    "threshold of ransac. 0.00025 for gcransac")
parser.add_argument("--obj_geod_th", type=float, default=1e-4, help=""
    "theshold for the good geodesic distance")


config = parser.parse_args()

def eval(data_loader, config):
    loader_iter = iter(data_loader)
    metrics = ['err_q', 'err_t', 'precision', 'recall', 'fscore']
    eval_res= {}
    t_net = 0
    t_est = 0

    for met in metrics:
        eval_res[met] = []
    for idx, test_data in enumerate(loader_iter):
        test_xs = test_data['xs'][0].numpy()
        x1, x2 = test_xs[0,:,:2], test_xs[0,:,2:4]
        K1, K2 = test_data['K1s'][0].numpy(), test_data['K2s'][0].numpy()
        dR, dt = test_data['Rs'][0].numpy(), test_data['ts'][0].numpy()
        y = test_data['ys'][0].numpy()
        gt = y[:,0] < config.obj_geod_th
        mask = np.logical_and((test_data['sides'][0,:,0].numpy() < config.ratio_test_th),
                              (test_data['sides'][0,:,1].numpy().astype(bool)))


        err_q, err_t, loss_q, loss_t, inliers, mask_updated, R, t = eval_decompose(x1, x2, dR, dt, mask=mask, ransac_type=config.ransac_type, ransac_th=config.ransac_th, K1=K1, K2=K2)
        precision = np.logical_and(mask_updated, gt).sum() / mask_updated.sum()
        recall = np.logical_and(mask_updated, gt).sum() / gt.sum()
        fscore = 2*precision*recall/(precision+recall+1e-15)
        print('idx '+str(idx)+' err_q '+str(err_q)+' err_t '+str(err_t)+' precision '+str(precision)+' recall '+str(recall)+' fscore '+str(fscore))
        eval_res['err_q'].append(err_q)
        eval_res['err_t'].append(err_t)
        eval_res['precision'].append(precision)
        eval_res['recall'].append(recall)
        eval_res['fscore'].append(fscore)
    ths, q_acc, t_acc, qt_acc = compute_metric(eval_res)
    for idx_th in range(1, len(ths)):
        print(str(ths[idx_th])+'deg: '+str(np.mean(qt_acc[:idx_th])))
    print('precision: ' +str(np.mean(np.asarray(eval_res['precision']))))
    print('recall: ' +str(np.mean(np.asarray(eval_res['recall']))))
    print('fscore: ' +str(np.mean(np.asarray(eval_res['fscore']))))

test_dataset = CorrespondencesDataset(config.data_te)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=False, collate_fn=collate_fn)

eval(test_loader, config)
