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
from torchcp.classification.utils import ConfCalibrator
from torchcp.classification.utils.metrics import Metrics
from torchcp.utils import fix_randomness
from torchcp.utils.common import get_device



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

# class Mixup(object):
#     def __init__(self, alpha=1.0):
#         self.alpha = alpha

#     def __call__(self, img, target):
#         lam = np.random.beta(self.alpha, self.alpha)
#         rand_index = torch.randperm(img.size()[0])

#         img = lam * img + (1 - lam) * img[rand_index, :]
#         target = lam * target + (1 - lam) * target[rand_index]
#         return img, target
    
def cal_acc(model, dataloader):
    device = get_device(model)
    model.eval()
    logits_transformation = ConfCalibrator.registry_ConfCalibrator("TS")(temperature=1)
    with torch.no_grad():
        acc_num = 0
        for examples in dataloader:
            tmp_x, tmp_labels = examples[0].to(device), examples[1].to(device)
            tmp_logits = logits_transformation(model(tmp_x)).detach()
            
            _, predicted = torch.max(tmp_logits, 1)
            acc_num += (predicted == tmp_labels).sum().item()
        
        accuracy = acc_num / len(dataloader)
    return accuracy
    
def test_imagenet(seed = 1, gpu = 0, csv_file_path = '/home/zhouxn/project/TorchCP/output/experiment_aug_method.csv'):
    fix_randomness(seed=seed)
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    model_name = 'ResNet101'
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = dset.ImageFolder("/data/dataset/imagenet/images/val", transforms)
    

    cal_dataset, test_dataset, _, _, _ = torch.utils.data.random_split(dataset, [10000, 10000, 10000, 10000, 10000])
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
        trn.AutoAugment(), # 自动增强
    ]
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Data', 'Model', 'Score', 'Predictor', 'Alpha', 'Augmentation Method', 'Seed', 'Threshold', 'Acc on calset', 'Coverage rate on calsete', 'Average size on calset', 'Acc on testset', 'Coverage rate on testset', 'Average size on testset']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        for score in score_functions:
            for class_predictor in predictors:
                for augmentation_method in augmentation_methods:
                    cal_dataset_aug = TransformedDataset(cal_dataset, augmentation_method)
                    cal_data_loader = DataLoader(cal_dataset_aug, batch_size=1024, shuffle=False, pin_memory=True)
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
    test_imagenet(seed = 0, gpu = 0,  csv_file_path = '/home/zhouxn/project/TorchCP/output/experiment_aug_method_seed_0.csv')
    # test_imagenet(seed = 1, gpu = 1, csv_file_path = '/home/zhouxn/project/TorchCP/output/experiment_aug_method_seed_1.csv')
    # test_imagenet(seed = 2)
    # test_imagenet(seed = 3)
    