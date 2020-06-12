'''Data Loader function in PyTorch.

Reference:
[1] Dense Depth https://github.com/ialhashim/DenseDepth/tree/master/PyTorch
'''

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
# import random
from pathlib import Path
import matplotlib.pyplot as plt
from EVA4.Modules.utils import colorize

bs   = 25
size = 256
root_folder = Path('./dataset/')

mean = [0.5704, 0.5221, 0.4675]
std  = [0.2504, 0.2552, 0.2709]

gen_transform = transforms.Compose([
                                      transforms.Resize((size,size)),
                                      transforms.ToTensor()
                                      ])

fg_bg_transform = transforms.Compose([
                                      transforms.Resize((size,size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,std)
                                      ])

class dataset:
    """
    Class to load the data from zip file into memory
    """

    def __init__(self):
        self.file_list,self.fg_bg_data,self.mask_data,self.depth_map_data = loadZipToMem()

    def __len__(self):
      return len(self.file_list)

class MasterDataset(Dataset):

    def __init__(self,dataset_obj,transform=True):
        self.file_list      = dataset_obj.file_list
        self.fg_bg_data     = dataset_obj.fg_bg_data
        self.mask_data      = dataset_obj.mask_data
        self.depth_map_data = dataset_obj.depth_map_data
        self.transform      = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):

        sample = self.file_list[idx]

        bg    = Image.open(root_folder/('bg/'+sample[0:5] + ".jpg"))
        fg_bg = Image.open(BytesIO(self.fg_bg_data['fg_bg/'+ sample]))
        mask  = Image.open(BytesIO(self.mask_data['mask/'+ sample.replace("jpg",'png')])).convert('RGB')
        depth = Image.open(BytesIO(self.depth_map_data['depth_map/'+ sample]) )

        if self.transform:
            bg     = fg_bg_transform(bg)
            fg_bg  = fg_bg_transform(fg_bg)

        else:
            bg     = gen_transform(bg)
            fg_bg  = gen_transform(fg_bg)
        mask   = (gen_transform(mask) > 0.8).float()
        depth  = gen_transform(depth)

        return {'bg': bg, 'fg_bg': fg_bg, 'mask': mask, 'depth': depth}


def loader(dataset_obj):
    # Defining CUDA
    cuda = torch.cuda.is_available()
    print("CUDA availability ?",cuda)

    # Defining data loaders with setting
    dataloaders_args = dict(shuffle=True, batch_size = bs, num_workers = 4, pin_memory = True) if cuda else dict(shuffle=True,batch_size = bs)

    train_ds = MasterDataset(dataset_obj,transform=True)

    return(DataLoader(train_ds, **dataloaders_args),DataLoader(train_ds, **dataloaders_args))

def sample_pictures(dataset_obj):

    sample_ds = MasterDataset(dataset_obj,transform=False)
    sample_dl = DataLoader(sample_ds, batch_size=4, shuffle=True)

    # get some random training images
    sample = next(iter(sample_dl))

    fig = plt.figure(figsize=(10, 10))

    imgs = sample['bg']

    grid_tensor = utils.make_grid(imgs, nrow=4)
    grid_image = grid_tensor.permute(1,2,0)

    ax = fig.add_subplot(4, 1, 1, xticks=[], yticks=[])
    ax.imshow(grid_image)

    imgs = sample['fg_bg']

    grid_tensor = utils.make_grid(imgs, nrow=4)
    grid_image = grid_tensor.permute(1,2,0)

    ax = fig.add_subplot(4, 1, 2, xticks=[], yticks=[])
    ax.imshow(grid_image)

    imgs = sample['mask']

    grid_tensor = utils.make_grid(imgs, nrow=4)
    grid_image = grid_tensor.permute(1,2,0)

    ax = fig.add_subplot(4, 1, 3, xticks=[], yticks=[])
    ax.imshow(grid_image)

    imgs = sample['depth']

    grid_tensor = colorize(utils.make_grid(imgs, nrow=4, normalize=False))
    grid_image = grid_tensor

    ax = fig.add_subplot(4, 1, 4, xticks=[], yticks=[])
    ax.imshow(grid_image)

    fig.tight_layout()  
    plt.show()

def loadZipToMem():
    # Load zip file into memory
    from zipfile import ZipFile
    
    fg_bg_zip = root_folder/'fg_bg.zip'
    mask_zip = root_folder/'mask.zip'
    depth_map_zip = root_folder/'depth_map.zip'

    input_zip = ZipFile(fg_bg_zip)
    fg_bg_data = {name: input_zip.read(name) for name in input_zip.namelist()}

    input_zip = ZipFile(mask_zip)
    mask_data = {name: input_zip.read(name) for name in input_zip.namelist()}

    input_zip = ZipFile(depth_map_zip)
    depth_map_data = {name: input_zip.read(name) for name in input_zip.namelist()}

    df = pd.read_csv(root_folder/'filelist.csv')
    file_list = list(df['fg_bg_filename'].to_list())

    from sklearn.utils import shuffle
    file_list = shuffle(file_list, random_state=0)

    #print(len(data))

    return file_list,fg_bg_data,mask_data,depth_map_data