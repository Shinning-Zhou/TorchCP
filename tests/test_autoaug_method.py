import argparse
import os
import pickle
import csv

from sklearn.metrics import accuracy_score

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from torch.utils.data import DataLoader, Dataset

from torchcp.classification.predictors import SplitPredictor, ClusterPredictor, ClassWisePredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS, Margin
from torchcp.classification.utils import ConfCalibrator
from torchcp.utils import fix_randomness
from torchcp.utils.common import get_device
from torchcp.classification.utils import ConfCalibrator




class TransformedDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, item):
        image, label = self.dataset[item]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

def cal_acc(model, cal_dataloader):
    model.eval()
    logits_list = []
    labels_list = []
    device = get_device(model)
    with torch.no_grad():
        for examples in cal_dataloader:
            tmp_x, tmp_labels = examples[0].to(device), examples[1].to(device)
            logits_transformation = ConfCalibrator.registry_ConfCalibrator("TS")(temperature=1)
            tmp_logits = logits_transformation(model(tmp_x)).detach()
            logits_list.append(tmp_logits)
            labels_list.append(tmp_labels)
        
        logits = torch.cat(logits_list).float()
        _, preds = torch.max(logits, 1)
        preds = preds.cpu().numpy()
        labels = torch.cat(labels_list).cpu().numpy()
        
    acc = accuracy_score(preds, labels)
    return acc

def No_transform(img):
    return img

pre_transforms = trn.Compose([trn.Resize(256, antialias=True),
                        trn.CenterCrop(224),
                        ])
                        
post_transforms = trn.Compose([
                        trn.ToTensor(),
                        trn.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                        ])


auto_augmentation_methods = [
    trn.AutoAugment(),
    trn.RandAugment(),
    trn.TrivialAugmentWide(),
    trn.AugMix(),
]
       
    
def test_imagenet(seed = 1, gpu = 0, csv_file_path = '/home/zhouxn/project/TorchCP/output/experiment_aug_method.csv'):
    fix_randomness(seed=seed)
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    model_name = 'ResNet101'
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = dset.ImageFolder("/data/dataset/imagenet/images/val", pre_transforms)
    
    # cal_dataset, test_dataset, _, _, _ = torch.utils.data.random_split(dataset, [10000, 10000, 10000, 10000, 10000])
    cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [25000, 25000])
    
    test_dataset_aug = TransformedDataset(test_dataset, post_transforms)
    test_data_loader = DataLoader(test_dataset_aug, batch_size=2048, shuffle=False, pin_memory=True)
    
    alpha = 0.1
    predictors = [SplitPredictor]
    score_functions = [APS()]
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Data', 'Model', 'Score', 'Predictor', 'Alpha', 'Augmentation Method', 'Seed', 'Threshold', 'Acc on calset', 'Coverage rate on calsete', 'Average size on calset', 'Acc on testset', 'Coverage rate on testset', 'Average size on testset']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        for score in score_functions:
            for class_predictor in predictors:
                for augmentation_method in auto_augmentation_methods:
                    cal_dataset_aug = TransformedDataset(cal_dataset, trn.Compose([augmentation_method, post_transforms]))
                    cal_data_loader = DataLoader(cal_dataset_aug, batch_size=2048, shuffle=False, pin_memory=True)
                    predictor = class_predictor(score, model)
                    predictor.calibrate(cal_data_loader, alpha)
                    
                    threshold = predictor.q_hat.item()
                    
                    acc_on_calset = cal_acc(model, cal_data_loader)
                    res_on_calset = predictor.evaluate(cal_data_loader)
                    
                    acc_on_testset = cal_acc(model, test_data_loader)
                    res_on_testset = predictor.evaluate(test_data_loader)
                    
                    print(f"Experiment Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}, Augmentation method : {augmentation_method.__class__.__name__}, Seed : {seed}")
                    print(f"Threshold : {threshold}, Acc on calset : {acc_on_calset}, Coverage rate on calset : {res_on_calset['Coverage_rate']}, Average size on calset : {res_on_calset['Average_size']}")
                    print(f"Acc on testset : {acc_on_testset}, Coverage rate on testset : {res_on_testset['Coverage_rate']}, Average size on testset : {res_on_testset['Average_size']}")
                    result_dict = {
                        'Data': 'ImageNet',
                        'Model': model_name,
                        'Score': score.__class__.__name__,
                        'Predictor': predictor.__class__.__name__,
                        'Alpha': alpha,
                        'Augmentation Method': augmentation_method.__class__.__name__,
                        'Seed': seed,
                        'Threshold': threshold,
                        'Coverage rate on calsete': res_on_calset['Coverage_rate'],
                        'Average size on calset': res_on_calset['Average_size'],
                        'Coverage rate on testset': res_on_testset['Coverage_rate'],
                        'Average size on testset': res_on_testset['Average_size'],
                        'Acc on calset': acc_on_calset, 
                        'Acc on testset': acc_on_testset,
                    }
                    csv_writer.writerow(result_dict)
                
if __name__ == '__main__':
    # test_imagenet(seed = 0, gpu = 0,  csv_file_path = '/home/zhouxn/project/TorchCP/output/experiment_autoaug_method_seed_0.csv')
    test_imagenet(seed = 1, gpu = 1, csv_file_path = '/home/zhouxn/project/TorchCP/output/experiment_autoaug_method_seed_1.csv')
    