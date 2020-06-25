import cv2
from transformations import quaternion_from_matrix
import numpy as np

def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # dR = np.dot(R, R_gt.T)
    # dt = t - np.dot(dR, t_gt)
    # dR = np.dot(R, R_gt.T)
    # dt = t - t_gt
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython
        IPython.embed()

    return err_q, err_t

def compute_metric(eval_res):
    ths = np.arange(7) * 5
    cur_err_q = np.array(eval_res["err_q"]) * 180.0 / np.pi
    cur_err_t = np.array(eval_res["err_t"]) * 180.0 / np.pi
    # Get histogram
    q_acc_hist, _ = np.histogram(cur_err_q, ths)
    t_acc_hist, _ = np.histogram(cur_err_t, ths)
    qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
    num_pair = float(len(cur_err_q))
    q_acc_hist = q_acc_hist.astype(float) / num_pair
    t_acc_hist = t_acc_hist.astype(float) / num_pair
    qt_acc_hist = qt_acc_hist.astype(float) / num_pair
    q_acc = np.cumsum(q_acc_hist)
    t_acc = np.cumsum(t_acc_hist)
    qt_acc = np.cumsum(qt_acc_hist)

    return ths, q_acc, t_acc, qt_acc


def eval_decompose(p1s, p2s, dR, dt, mask=None, probs=None,
                   weighted=False, use_prob=True, ransac_th=1e-4, ransac_type=0, K1=None, K2=None):
    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    # Mask the ones that will not be used
    p1s_good = p1s[mask]
    p2s_good = p2s[mask]
    probs_good = None
    if probs is not None:
        probs_good = probs[mask]

    num_inlier = 0
    mask_new2 = None
    R, t = None, None
    if p1s_good.shape[0] >= 5:
        if probs is None:
            # Change the threshold from 0.01 to 0.001 can largely imporve the results
            # For fundamental matrix estimation evaluation, we also transform the matrix to essential matrix.
            # This gives better results than using findFundamentalMat
            if ransac_type == 1:
                import pymagsac
                E, mask_new = pymagsac.findEssentialMatrix(p1s_good, p2s_good, K1=np.eye(3), K2=np.eye(3), sigma_th=ransac_th, minimum_inlier_ratio_in_validity_check=0.1)
            elif ransac_type == 2:
                import pygcransac
                # back to pixel coordinate for gcransac
                p1s_coord = p1s_good * np.asarray([K1[0,0], K1[1,1]]) + np.array([K1[0,2], K1[1,2]])
                p2s_coord = p2s_good * np.asarray([K2[0,0], K2[1,1]]) + np.array([K2[0,2], K2[1,2]])
                E, mask_new = pygcransac.findEssentialMatrix(p1s_coord, p2s_coord, K1, K2, int(K1[1,2]*2), int(K1[0,2]*2), int(K2[1,2]*2), int(K2[0,2]*2), ransac_th, 0.999999, max_iters=10000)
            elif ransac_type == 0:
                E, mask_new = cv2.findEssentialMat(p1s_good, p2s_good, method=cv2.RANSAC, threshold=0.001)

        else:
            pass
        if E is not None:
            new_RT = False
            # Get the best E just in case we get multipl E from
            # findEssentialMat
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _mask_new2 = cv2.recoverPose(
                    _E, p1s_good, p2s_good, mask=mask_new.reshape(-1,1).astype(np.uint8))
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    mask_new2 = _mask_new2
                    new_RT = True
            if new_RT:
                err_q, err_t = evaluate_R_t(dR, dt, R, t)
            else:
                err_q = np.pi
                err_t = np.pi / 2

        else:
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new2 is not None:
        # Change mask type
        mask_new2 = mask_new2.flatten().astype(bool)
        mask_updated[mask] = mask_new2

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated, R, t

