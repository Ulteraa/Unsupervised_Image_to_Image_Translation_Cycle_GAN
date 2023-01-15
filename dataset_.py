import torch
import torch.nn as nn
import torchvision
from torch.utils.data import  DataLoader , Dataset
from PIL import  Image
from torchvision import transforms
import os
class translation_dataset(Dataset):
    def __init__(self, root_s, root_t):
        super(translation_dataset, self).__init__()
        self.root_t=root_t
        self.root_s=root_s
        self.list_source=os.listdir(self.root_s)
        self.list_target=os.listdir(self.root_t)
        self.target_len=len(self.list_target)
        self.source_len = len(self.list_source)
    def __len__(self):
        return max(len(self.list_source),len(self.list_target))

    def __getitem__(self, item):
        source=self.list_source[item%self.source_len]
        target=self.list_target[item%self.target_len]
        source_img=Image.open(os.path.join(self.root_s,source)).convert('RGB')
        target_img = Image.open(os.path.join(self.root_t, target)).convert('RGB')
        transfrom=transforms.Compose([transforms.Resize((256,256)),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), transforms.ToTensor()])
        source_img=transfrom(source_img)
        target_img=transfrom(target_img)
        return source_img, target_img

