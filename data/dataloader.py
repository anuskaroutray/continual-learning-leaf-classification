import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset
import pandas as pd

from PIL import Image

from torchvision import transforms
import numpy as np


class DeepHerbDataset(Dataset):

    def __init__(self, data_dir, num_classes = None, mode = 'train'):
        super(DeepHerbDataset, self).__init__()

        self.data_dir = data_dir

        class_dir_list = os.listdir(self.data_dir)

        self.num_classes = len(class_dir_list)

        img_path_list = []
        img_labels_list = []
        for label_idx, class_dir in enumerate(class_dir_list):

            l = os.listdir(os.path.join(self.data_dir, class_dir))
            img_path_list.extend([os.path.join(self.data_dir, class_dir, li) for li in l])
            img_labels_list.extend([label_idx] * len(l))

        data_dict = {"image_path": img_path_list,
                        "label": img_labels_list}
        
        self.df = pd.DataFrame(data_dict).sample(frac = 1)
        if mode == "train":
            self.df = self.df[:int(0.8 * len(self.df))]
            self.data_path = os.path.join(self.data_dir, f"{mode}-leaf-data.csv")
        else:
            self.df = self.df[int(0.8 * len(self.df)):]
            self.data_path = os.path.join(self.data_dir, f"{mode}-leaf-data.csv")

        self.df.reset_index(inplace = True)
        
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                             transforms.CenterCrop(224)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = np.array(Image.open(self.df["image_path"][idx], mode = "r"), dtype = np.float32)
        image = self.transform(image)
        label = self.df["label"][idx]

        # TODO: Add data augmentation
        return image, label
