from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np
import json

class CUB200(iData):
    '''
    Dataset Name:   CUB200-2011
    Task:           fine-grain birds classification
    Data Format:    224x224 color images. (origin imgs have different w,h)
    Data Amount:    5,994 images for training and 5,794 for validationg/testing
    Class Num:      200
    Label:          

    Reference:      https://opendatalab.com/CUB-200-2011
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = 224 if img_size is None else img_size
        self.train_trsf = [
            transforms.RandomResizedCrop(224, (0.6, 1)),
            transforms.RandomRotation(20),
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
        self.test_trsf = []

        
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(200).tolist()

    def getdata(self, train:bool, root_dir, img_dir):
        data, targets = [], []
        with open(os.path.join(root_dir, 'train_test_split.txt')) as f:
            for line in f:
                image_id, is_train = line.split()
                if int(is_train) == int(train):
                    data.append(os.path.join(img_dir, self.images_path[image_id]))
                    targets.append(self.class_ids[image_id])
            
        return np.array(data), np.array(targets)

    def download_data(self):
        root_dir = os.path.join(os.environ["DATA"], 'CUB_200_2011')
        img_dir = os.path.join(root_dir, 'images')

        self.images_path = {}
        with open(os.path.join(root_dir, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(root_dir, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = int(class_id) - 1

        self.train_data, self.train_targets = self.getdata(True, root_dir, img_dir)
        self.test_data, self.test_targets = self.getdata(False, root_dir, img_dir)

        # print(len(np.unique(self.train_targets))) # output: 200
        # print(len(np.unique(self.test_targets))) # output: 200

        self.get_class_name(txt_path=os.path.join(root_dir, "classes.txt"))
        self.templates = None
        
    def get_class_name(self, txt_path):
        class_name = ['None'] * (200)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            target, name = line.split(' ')
            cls_name = name.split('.')[1].replace('_', ' ')
            if class_name[int(target)-1] == 'None':
                class_name[int(target)-1] = cls_name
        self.classes_name = class_name

    def get_tempates_json(self, json_path):
        with open(json_path, "r") as f:
            templates = json.load(f)
        self.templates = templates