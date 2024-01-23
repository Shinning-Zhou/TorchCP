import os
import argparse
from tqdm import tqdm

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
import torchvision.models as models
import torchvision.transforms as trn
import torchvision.datasets as dset

from torchcp.classification.predictors import SplitPredictor, ClusterPredictor, ClassWisePredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS, Margin
from torchcp.utils import fix_randomness


def build_transfer_dataloader(dataset_name, transform=None, mode='train'):
    #  path of usr
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data/dataset")
    
    if transform is None:
        transform = trn.Compose([
            trn.Resize(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
    
    if dataset_name == 'imagenet':
        dataset = dset.ImageFolder(data_dir + "/imagenet/val", transform)
    elif dataset_name == 'imagenetv2':
        dataset = dset.ImageFolder(data_dir + "/imagenetv2-matched-frequency-format-val", transform)
    elif dataset_name == 'imagenet-sketch':
        dataset = dset.ImageFolder(data_dir + "/imagenet-sketch/images", transform)
    elif dataset_name == 'imagenet-r':
        dataset = dset.ImageFolder(data_dir + "/imagenet-rendition/imagenet-r", transform)
    else:
        raise NotImplementedError

    return dataset


def get_logits(model, dataset, device, num_classes: int = 1000, batch_size: int = 128, num_workers: int = 4):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    logits = torch.zeros((len(dataset), num_classes), dtype=torch.float)
    labels = torch.zeros((len(dataset),), dtype=torch.int)
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(dataloader)):
            input = input.to(device)
            target = target.to(device)

            output = model(input)

            logits[i * batch_size : i * batch_size + input.shape[0],:] = output.detach().cpu()
            labels[i * batch_size : i * batch_size + input.shape[0]] = target.detach().cpu()

    return torch.utils.data.TensorDataset(logits, labels)

if __name__ == '__main__':
    ##################################
    # Preparing dataset
    ##################################
    cuda = 0
    device = torch.device("cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    fix_randomness(seed=0)
    
    # dataset name
    val_dataset_name = 'imagenet'
    test_dataset_name = 'imagenetv2'
    
    # pre-trained classifier
    model_name = 'resnet50'
    model = models.resnet50(weights="IMAGENET1K_V1", progress=True)
    model.to(device)
    
    # data params
    batch_size = 128
    num_workers = 4
    
    # dataset
    # val_dataset = get_dataset(val_dataset_name)
    # val_logits = get_logits(model, val_dataset, batch_size = batch_size, num_workers = num_workers, device = device)
    # print(val_dataset)
    # print(val_logits)
    # cur_dataset = get_dataset(cur_dataset_name)
    # cur_logits = get_logits(model, cur_dataset, batch_size = batch_size, num_workers = num_workers, device = device)
    cal_data_loader = build_transfer_dataloader(val_dataset_name)
    test_data_loader = build_transfer_dataloader(test_dataset_name)
    
    #######################################
    # A standard process of conformal prediction
    #######################################    
    alpha = 0.1
    print(f"Calibration data: {val_dataset_name}, Test data: {test_dataset_name}")
    score_function = THR()
    conformal_predictor = SplitPredictor(score_function, model)
    
    conformal_predictor.calibrate(cal_data_loader, alpha)
    result = conformal_predictor.evaluate(test_data_loader)
    print(f"Result--Coverage_rate: {result['Coverage_rate']}, Average_size: {result['Average_size']}")