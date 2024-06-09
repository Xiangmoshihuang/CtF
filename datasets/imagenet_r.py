from torchvision import datasets, transforms
from datasets.idata import iData
import os
import numpy as np
from utils.toolkit import split_images_labels
import json

class ImageNet_R(iData):
    '''
    Dataset Name:   ImageNet_R dataset
    Source:         A subset of the Tiny Images dataset.
    Task:           Classification Task
    Data Format:    32x32 color images.
    Data Amount:    60000 (500 training images and 100 testing images per class)
    Class Num:      100 (grouped into 20 superclass).
    Label:          Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = img_size if img_size != None else 224
        self.train_trsf = [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
        ]
        self.test_trsf = [
            transforms.Resize((256,256)), # 256
            transforms.CenterCrop(224), # 224
        ]
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.230, 0.227, 0.226]),
        ]
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        root_dir = os.environ['DATA']

        train_dataset = datasets.ImageFolder(os.path.join(os.environ['DATA'], 'imagenet_r_old', 'train'))
        test_dataset = datasets.ImageFolder(os.path.join(os.environ['DATA'], 'imagenet_r_old', 'test'))
        
        self.train_data, self.train_targets = split_images_labels(train_dataset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dataset.imgs)
        self.get_class_name(os.path.join(root_dir, 'imagenet_r', 'ir_classes.txt'))
        self.get_tempates_json(os.path.join("datasets","all_prompts","imagenet_prompts_base.json"))

    def get_class_name(self, txt_path):
        class_name = ['None'] * (200)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for idx, line in enumerate(lines):
            name = line.split('\n')[0]
            cls_name = name[9:]
            if class_name[idx] == 'None':
                class_name[idx] = cls_name
        self.classes_name = class_name



    def get_tempates_json(self, json_path):
        with open(json_path, "r") as f:
            templates = json.load(f)
        self.templates = templates