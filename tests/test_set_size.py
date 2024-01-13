import argparse
import os
import pickle
import csv

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from torch.utils.data import DataLoader, Dataset

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
        
        
def No_transform(img):
    return img
    
def test_imagenet():
    fix_randomness(seed=0)
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    model_name = 'ResNet101'
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = dset.ImageFolder("/data/dataset/imagenet/images/val", transforms)
    
    cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [40000, 10000])
    test_data_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True)
    
    alpha = 0.1
    
    predictors = [SplitPredictor]
    score_functions = [APS()]
    augmentation_methods = [
        trn.Lambda(No_transform), # 不做变化
        # Affine transformations
        trn.RandomVerticalFlip(),  # 随机垂直翻转
        trn.RandomHorizontalFlip(),  # 随机水平翻转
        trn.RandomRotation(degrees=45),  # 随机旋转，角度范围为[-45, 45]
        trn.RandomCrop(size=(224, 224)),  # 随机裁剪
        trn.RandomResizedCrop(size=(224, 224)),  # 随机裁剪和缩放
        trn.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 随机仿射变换
        # Elastic transformations
        trn.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色扭曲
        trn.RandomGrayscale(),  # 随机灰度化
        trn.RandomPerspective(), # 随机透视
        trn.ElasticTransform(), # 弹性变化
        trn.RandomAutocontrast(), # 随机调整图像的对比度
        trn.RandomAdjustSharpness(sharpness_factor=0.5), # 随机减少图像的锐度
        trn.RandomAdjustSharpness(sharpness_factor=1.5), # 随机增加图像的锐度
        trn.GaussianBlur(kernel_size=5), # 高斯模糊
        # Advanced transformations
        trn.RandomErasing(),  # 随机擦除
        
        # Add Mixup if needed
        # Mixup(alpha=0.2, p=0.5),  # Mixup混合
    ]
    cal_sizes = [20000, 30000, 40000]
    csv_file_path = '/home/zhouxn/project/TorchCP/output/experiment_set_size.csv'
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Data', 'Model', 'Score', 'Predictor', 'Alpha', 'Augmentation Method', 'Coverage rate', 'Average size', 'Cal_test_ratio']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        for score in score_functions:
            for class_predictor in predictors:
                for augmentation_method in augmentation_methods:
                    for cal_size in cal_sizes:
                        cal_dataset_cut, _ = torch.utils.data.random_split(cal_dataset, [cal_size, len(cal_dataset) - cal_size])
                        cal_dataset_aug = TransformedDataset(cal_dataset_cut, augmentation_method)
                        cal_data_loader = DataLoader(cal_dataset_aug, batch_size=1024, shuffle=False, pin_memory=True)
                        predictor = class_predictor(score, model)
                        predictor.calibrate(cal_data_loader, alpha)
                        print(f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}, Augmentation method : {augmentation_method.__class__.__name__}")
                        res_dict = predictor.evaluate(test_data_loader)
                        print(res_dict)
                        result_dict = {
                            'Data': 'ImageNet',
                            'Model': model_name,
                            'Score': score.__class__.__name__,
                            'Predictor': predictor.__class__.__name__,
                            'Alpha': alpha,
                            'Augmentation Method': augmentation_method.__class__.__name__,
                            'Coverage rate': res_dict['Coverage_rate'],
                            'Average size': res_dict['Average_size'],
                            'Cal_test_ratio': cal_size / 10000, 
                        }
                        csv_writer.writerow(result_dict)
                

if __name__ == '__main__':
    test_imagenet()
    