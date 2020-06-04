'''Data Loader function in PyTorch.

Reference:
[1] Dense Depth https://github.com/ialhashim/DenseDepth/tree/master/PyTorch
'''

import pandas as pd
import numpy as np
# import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
# import random
from pathlib import Path
import matplotlib.pyplot as plt

size = 256
root_folder = Path('./dataset/')

gen_transform = transforms.Compose([
                                      transforms.Resize((size,size)),
                                      transforms.ToTensor()
                                      ])

class dataset:
    """
    Class to load the data and define the data loader
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
            bg = gen_transform(bg)
            fg_bg = gen_transform(fg_bg)
            mask = (gen_transform(mask) > 0.8).float()
            depth = gen_transform(depth)

        return {'bg': bg, 'fg_bg': fg_bg, 'mask': mask, 'depth': depth}

def sample_pictures(master_dataset_obj):

    train_dl = DataLoader(master_dataset_obj, batch_size=4, shuffle=True)

    # get some random training images
    sample = next(iter(train_dl))

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

    grid_tensor = utils.make_grid(imgs, nrow=4)
    grid_image = grid_tensor.permute(1,2,0)

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

# def _is_pil_image(img):
#     return isinstance(img, Image.Image)

# def _is_numpy_image(img):
#     return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

# class RandomHorizontalFlip(object):
#     def __call__(self, sample):
#         image, depth = sample['image'], sample['depth']

#         if not _is_pil_image(image):
#             raise TypeError(
#                 'img should be PIL Image. Got {}'.format(type(image)))
#         if not _is_pil_image(depth):
#             raise TypeError(
#                 'img should be PIL Image. Got {}'.format(type(depth)))

#         if random.random() < 0.5:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#             depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

#         return {'image': image, 'depth': depth}

# class RandomChannelSwap(object):
#     def __init__(self, probability):
#         from itertools import permutations
#         self.probability = probability
#         self.indices = list(permutations(range(3), 3))

#     def __call__(self, sample):
#         image, depth = sample['image'], sample['depth']
#         if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
#         if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
#         if random.random() < self.probability:
#             image = np.asarray(image)
#             image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
#         return {'image': image, 'depth': depth}

# def loadZipToMem(zip_file):
#     # Load zip file into memory
#     print('Loading dataset zip file...', end='')
#     from zipfile import ZipFile
#     input_zip = ZipFile(zip_file)
#     data = {name: input_zip.read(name) for name in input_zip.namelist()}
#     nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))

#     from sklearn.utils import shuffle
#     nyu2_train = shuffle(nyu2_train, random_state=0)

#     #if True: nyu2_train = nyu2_train[:40]

#     print('Loaded ({0}).'.format(len(nyu2_train)))
#     return data, nyu2_train

# class depthDatasetMemory(Dataset):
#     def __init__(self, data, nyu2_train, transform=None):
#         self.data, self.nyu_dataset = data, nyu2_train
#         self.transform = transform

#     def __getitem__(self, idx):
#         sample = self.nyu_dataset[idx]
#         image = Image.open( BytesIO(self.data[sample[0]]) )
#         depth = Image.open( BytesIO(self.data[sample[1]]) )
#         sample = {'image': image, 'depth': depth}
#         if self.transform: sample = self.transform(sample)
#         return sample

#     def __len__(self):
#         return len(self.nyu_dataset)

# class ToTensor(object):
#     def __init__(self,is_test=False):
#         self.is_test = is_test

#     def __call__(self, sample):
#         image, depth = sample['image'], sample['depth']
        
#         image = self.to_tensor(image)

#         depth = depth.resize((320, 240))

#         if self.is_test:
#             depth = self.to_tensor(depth).float() / 1000
#         else:            
#             depth = self.to_tensor(depth).float() * 1000
        
#         # put in expected range
#         depth = torch.clamp(depth, 10, 1000)

#         return {'image': image, 'depth': depth}

#     def to_tensor(self, pic):
#         if not(_is_pil_image(pic) or _is_numpy_image(pic)):
#             raise TypeError(
#                 'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

#         if isinstance(pic, np.ndarray):
#             img = torch.from_numpy(pic.transpose((2, 0, 1)))

#             return img.float().div(255)

#         # handle PIL Image
#         if pic.mode == 'I':
#             img = torch.from_numpy(np.array(pic, np.int32, copy=False))
#         elif pic.mode == 'I;16':
#             img = torch.from_numpy(np.array(pic, np.int16, copy=False))
#         else:
#             img = torch.ByteTensor(
#                 torch.ByteStorage.from_buffer(pic.tobytes()))
#         # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
#         if pic.mode == 'YCbCr':
#             nchannel = 3
#         elif pic.mode == 'I;16':
#             nchannel = 1
#         else:
#             nchannel = len(pic.mode)
#         img = img.view(pic.size[1], pic.size[0], nchannel)

#         img = img.transpose(0, 1).transpose(0, 2).contiguous()
#         if isinstance(img, torch.ByteTensor):
#             return img.float().div(255)
#         else:
#             return img

# def getNoTransform(is_test=False):
#     return transforms.Compose([
#         ToTensor(is_test=is_test)
#     ])

# def getDefaultTrainTransform():
#     return transforms.Compose([
#         RandomHorizontalFlip(),
#         RandomChannelSwap(0.5),
#         ToTensor()
#     ])

# def getTrainingTestingData(batch_size):
#     data, nyu2_train = loadZipToMem('nyu_data.zip')

#     transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
#     transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())

#     return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size, shuffle=False)
