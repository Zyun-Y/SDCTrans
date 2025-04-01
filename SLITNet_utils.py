# SLIT-Net
# DOI: 10.1109/JBHI.2020.2983549

import os
import multiprocessing
import h5py as hh
import numpy as np
import csv
import skimage.io
import skimage.transform
import skimage.segmentation
import skimage.measure
from skimage.measure import find_contours
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial.distance import directed_hausdorff

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from matplotlib import patches
from matplotlib.patches import Polygon


def calculate_hausdorff_metric(pred, truth, num_classes):

    # Ensure the dimensions are the same:
    # if not (pred["masks"].shape[0:2] == truth["masks"].shape[0:2]):
    #     print('Performance statistics cannot be calculated because dimensions do not match.')
    #     return

    # Classes:
    classes = range( num_classes )
    class_hausdorff = np.zeros([num_classes, 1])

    # Ensure masks are boolean:
    pred = pred.cpu().numpy().astype(bool)
    truth = truth.cpu().numpy().astype(bool)

    for c in classes:

        # # Initialise:
        # truth_mask = np.zeros_like(truth["masks"][:, :, 0])
        # pred_mask = np.zeros_like(truth_mask)

        # # Get indices:
        # pred_idxs = np.where(pred["class_ids"] == c)[0]
        # truth_idxs = np.where(truth["class_ids"] == c)[0]

        # # Get binary maps:
        # for i in pred_idxs:
        #     pred_mask += pred["masks"][:, :, i]

        pred_mask = pred[0, c]
        pred_mask = pred_mask.astype(bool)

        # for i in truth_idxs:
        #     truth_mask += truth["masks"][:, :, i]
        truth_mask = truth[0, c]
        truth_mask = truth_mask.astype(bool)
        # print(truth_mask.shape, pred_mask.shape)
        # Get contours to calculate HD:
        truth_contours = skimage.measure.find_contours(truth_mask, 0.5)
        pred_contours = skimage.measure.find_contours(pred_mask, 0.5)
        nTruth = len(truth_contours)
        nPred = len(pred_contours)

        # If there's only one contour for each then it's fine, just calculate:
        if nTruth == 1 and nPred == 1:
            tc = truth_contours[0]
            pc = pred_contours[0]
            hd = max(directed_hausdorff(tc, pc)[0], directed_hausdorff(pc, tc)[0])

        # But if there are multiple instances of each class, find the best match,
        # and those with no matches will not contribute
        # e.g. if there is an extra prediction, it does not contribute
        # Average the HD for all matches for that class
        elif nTruth > 0 and nPred > 0:
            sum_hd = 0
            sum_cnt = 0
            invalid_idxs = []
            for it in range(nTruth):
                # Compare each truth to all preds:
                tc = truth_contours[it]
                hd_all = np.zeros([nPred])
                for ip in range(nPred):
                    # Check if this pred has been matched with something else:
                    if ip in invalid_idxs:
                        hd_all[ip] = np.inf
                    else:
                        pc = pred_contours[ip]
                        hd_all[ip] = max(directed_hausdorff(tc, pc)[0], directed_hausdorff(pc, tc)[0])
                    # Find the best match:
                    best_idx = np.argmin(hd_all)
                    best_val = hd_all[best_idx]
                    # Add to sum:
                    if not best_val == np.inf:
                        sum_hd += best_val
                        sum_cnt += 1
                        # Remove this index from future comparisons:
                        invalid_idxs.append(best_idx)
            # Average per class:
            hd = sum_hd / sum_cnt

        elif nTruth == 0 and nPred == 0:
            hd = 0

        else:
            hd = np.nan

        # Update:
        class_hausdorff[c] = hd

    return class_hausdorff

