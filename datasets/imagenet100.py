from torchvision import transforms
from utils import myTransforms
from datasets.idata import iData
import os
import numpy as np
import json

class ImageNet100(iData):
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = img_size if img_size != None else 224
        self.train_trsf = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([myTransforms.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ]
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        self.test_trsf = [
            transforms.Resize((256,256)), # 256
            transforms.CenterCrop(224), # 224
        ]
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(100).tolist()

    def getdata(self, root_dir, fn):
        print('Opening '+fn)
        source_path = os.path.join(root_dir, 'ilsvrc2012')
        file = open(os.path.join(root_dir, 'imagenet_100', fn))
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                data.append(os.path.join(source_path, temp[0]))
                targets.append(int(temp[1]))
        return np.array(data), np.array(targets)

    def download_data(self):
        # assert 0,"You should specify the folder of your dataset"
        root_dir = os.environ['DATA']

        self.train_data, self.train_targets = self.getdata(root_dir, 'train_100.txt')
        self.test_data, self.test_targets = self.getdata(root_dir, 'val_100.txt')
        self.get_class_name(os.path.join(root_dir, 'imagenet_100', 'imagenet100_classes.txt'))
        self.get_tempates_json(os.path.join("datasets","all_prompts","imagenet_prompts_base.json"))


    def get_class_name(self, txt_path):
        class_name = ['None'] * (100)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            target, _, name = line.split('\t')
            cls_name = name.split('\n')[0]
            if class_name[int(target)] == 'None':
                class_name[int(target)] = cls_name
        self.classes_name = class_name

    def get_tempates_json(self, json_path):
        with open(json_path, "r") as f:
            templates = json.load(f)
        self.templates = templates