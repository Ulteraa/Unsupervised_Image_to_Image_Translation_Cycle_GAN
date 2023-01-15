import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import torch
lr_=1e-5
batch_size=1
root_s='maps/trainA'
root_t='maps/trainB'
root_v_s='maps/valA'
root_v_t='maps/valB'

lambda_=10
epochs=500
image_chanel=3
image_size=600
# chanel_dim_dis=[64,128,256,512]
# chanel_dim_gen=64
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
# transform_to_both= transforms.Compose([A. Resize(width=256,height=256), A.HorizontalFlip(0.05)],additional_targets={'image0:image'})
#
# transform_to_input=A.Compose([A.ColorJitter(0.1), A. Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]), ToTensorV2()])
# transform_to_output=A.Compose([A. Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]), ToTensorV2()])
