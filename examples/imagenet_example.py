# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
from pathlib import Path
sys.path.append(str(Path(__name__).resolve().parent))

import argparse
import os

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from tqdm import tqdm

from torchcp.classification.predictors import ClusterPredictor, ClassWisePredictor, SplitPredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS
from torchcp.classification import Metrics
from torchcp.utils import fix_randomness
from examples.common.dataset import build_dataset

from torchcp.utils.common import get_device
def cal_minimum(dataloader, model):
    device = get_device(model)
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for examples in dataloader:
            tmp_x, tmp_labels = examples[0].to(device), examples[1].to(device)
            tmp_logits = model(tmp_x).detach()
            logits_list.append(tmp_logits)
            labels_list.append(tmp_labels)
        logits = torch.cat(logits_list).float()
        labels = torch.cat(labels_list).view(-1, 1)  
        sorted_indices = torch.argsort(logits, descending=True)
        original_indices = torch.argsort(sorted_indices, dim=1)
        positions = original_indices.gather(1, labels)
        minimum = torch.mean(positions.float())+1
    breakpoint()
    return minimum
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    args = parser.parse_args()

    fix_randomness(seed=args.seed)

    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    # model_name = 'ResNet101'
    model_name = 'ResNet50'
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1", progress=True)
    model_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(model_device)


    dataset = build_dataset('imagenet')

    cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [25000, 25000])
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers = 4)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers = 4)

    # cal_minimum(test_data_loader, model)
    
    #######################################
    # A standard process of conformal prediction
    #######################################    
    alpha = args.alpha
    for score_function in [APS(), RAPS(0.01, 2), SAPS(0.1)]:
        print(f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score_function.__class__.__name__}, Predictor : SplitPredictor, Alpha : {alpha}")
        predictor = SplitPredictor(score_function, model)
        # print(f"The size of calibration set is {len(cal_dataset)}.")
        predictor.calibrate(cal_data_loader, alpha)
        print(predictor.evaluate(test_data_loader))
        

