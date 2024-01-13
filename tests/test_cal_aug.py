import argparse
import os
import pickle
import csv
import random

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import ConcatDataset

from torchcp.classification.predictors import SplitPredictor, ClusterPredictor, ClassWisePredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS, Margin
from torchcp.classification.utils.metrics import Metrics
from torchcp.utils import fix_randomness



transforms = trn.Compose([trn.Resize(256),
                        trn.CenterCrop(224),
                        trn.ToTensor(),
                        trn.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                        ])
       
class RandomTransformedDataset(Dataset):
    def __init__(self, dataset, transforms_list=None):
        self.dataset = dataset
        self.transforms_list = transforms_list

    def __getitem__(self, item):
        image, label = self.dataset[item]
        if self.transforms_list is not None:
            selected_transform = random.choice(self.transforms_list)
            image = selected_transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)
    
    
def No_transform(img):
    return img


def cal_time_aug(dataset, transform_list, n):
    # 返回一个(n+1)*len(dataset)的dataset
    dataset_res = dataset
    for _ in range(n):
        dataset_aug = RandomTransformedDataset(dataset, transform_list)
        dataset_res = dataset_res + dataset_aug
    return dataset_res

    
def test_imagenet():
    fix_randomness(seed=0)
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    model_name = 'ResNet101'
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = dset.ImageFolder("/data/dataset/imagenet/images/val", transforms).to(device)
    

    cal_dataset, test_dataset, _ = torch.utils.data.random_split(dataset, [5000, 5000, 40000])
    test_data_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True)


    alpha = 0.1
    predictors = [SplitPredictor]
    score_functions = [APS()]
    
    augmentation_methods = [
        # trn.Lambda(No_transform), # 不做变化
        # Affine transformations
        trn.RandomVerticalFlip(p=1),  # 随机垂直翻转
        trn.RandomHorizontalFlip(p=1),  # 随机水平翻转
        trn.RandomRotation(degrees=45),  # 随机旋转，角度范围为[-45, 45]
        trn.RandomCrop(size=(224, 224)),  # 随机裁剪
        trn.RandomResizedCrop(size=(224, 224)),  # 随机裁剪和缩放
        trn.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 随机仿射变换
        # Elastic transformations
        trn.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色扭曲
        trn.RandomGrayscale(p=1),  # 随机灰度化
        trn.RandomPerspective(p=1), # 随机透视
        trn.ElasticTransform(), # 弹性变化
        trn.RandomAutocontrast(p=1), # 随机调整图像的对比度
        trn.RandomAdjustSharpness(p=1, sharpness_factor=0.5), # 随机减少图像的锐度
        trn.RandomAdjustSharpness(p=1, sharpness_factor=1.5), # 随机增加图像的锐度
        trn.GaussianBlur(kernel_size=5), # 高斯模糊
        # Advanced transformations
        trn.RandomErasing(p=1),  # 随机擦除
        
        # Add Mixup if needed
        # Mixup(alpha=0.2, p=0.5),  # Mixup混合
    ]
    csv_file_path = '/home/zhouxn/project/TorchCP/output/experiment_cal_time_aug.csv'
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Data', 'Model', 'Score', 'Predictor', 'Alpha', 'Aug_method_num', 'Cal_data_ratio', 'Coverage rate', 'Average size']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        for score in score_functions:
            for class_predictor in predictors:
                for n in range(1, 5):
                    cal_dataset_aug = cal_time_aug(cal_dataset, augmentation_methods, n)
                    cal_data_loader = DataLoader(cal_dataset_aug, batch_size=1024, shuffle=False, pin_memory=True)
                    predictor = class_predictor(score, model)
                    predictor.calibrate(cal_data_loader, alpha)
                    
                    print(f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Aug_method_num : {n}, Cal_data_ratio : {n+1}, Alpha : {alpha}")
                    res_dict = predictor.evaluate(test_data_loader)
                    print(res_dict)
                    result_dict = {
                        'Data': 'ImageNet',
                        'Model': model_name,
                        'Score': score.__class__.__name__,
                        'Predictor': predictor.__class__.__name__,
                        'Alpha': alpha,
                        'Aug_method_num': n,
                        'Cal_data_ratio': n+1,
                        'Coverage rate': res_dict['Coverage_rate'],
                        'Average size': res_dict['Average_size'],
                    }
                    csv_writer.writerow(result_dict)
                    print(result_dict)
                
if __name__ == '__main__':
    test_imagenet()
    