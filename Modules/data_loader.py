# Torch libraries
import torch                   #PyTorch base libraries
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2

from albumentations import Compose, PadIfNeeded, RandomCrop,RandomBrightnessContrast, GaussianBlur, Normalize, HorizontalFlip, Resize, Cutout, ShiftScaleRotate,HueSaturationValue
from albumentations.pytorch import ToTensor

class album_Compose_train():
    def __init__(self):
        self.albumentations_transform = Compose([
            PadIfNeeded(40,40),
            RandomCrop(32,32),
            HorizontalFlip(),
            #RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5),
            #GaussianBlur(blur_limit=3, p=0.25),
            #ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.30, rotate_limit=45, p=.35),
            Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=[0.4914*255, 0.4822*255, 0.4465*255], always_apply=True, p=1.00),
            Normalize(mean=[0.4914, 0.4822, 0.4465],std=[.2023, 0.1994, 0.2010]),
            ToTensor()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class album_Compose_test():
    def __init__(self):
        self.albumentations_transform = Compose([
            Normalize(mean=[0.4914, 0.4822, 0.4465],std=[.2023, 0.1994, 0.2010]),
            ToTensor()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class dataset_cifar10:
    """
    Class to load the data and define the data loader
    """

    def __init__(self, batch_size=128):

        # Defining CUDA
        cuda = torch.cuda.is_available()
        print("CUDA availability ?",cuda)

        # Defining data loaders with setting
        self.dataloaders_args = dict(shuffle=True, batch_size = batch_size, num_workers = 4, pin_memory = True) if cuda else dict(shuffle=True,batch_size = batch_size)
        self.sample_dataloaders_args = self.dataloaders_args.copy()

        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def data(self, train_flag):

        # Transformations data augmentation (only for training)
        if train_flag :
            return datasets.CIFAR10('./Data',
                            train=train_flag,
                            transform=album_Compose_train(),
                            download=True)

        # Testing transformation - normalization adder
        else:
            return datasets.CIFAR10('./Data',
                                train=train_flag,
                                transform=album_Compose_test(),
                                download=True)

    # Dataloader function
    def loader(self, train_flag=True):
        return(torch.utils.data.DataLoader(self.data(train_flag), **self.dataloaders_args))


    def data_summary_stats(self):
        # Load train data as numpy array
        train_data = self.data(train_flag=True).data
        test_data = self.data(train_flag=False).data

        total_data = np.concatenate((train_data, test_data), axis=0)
        print(total_data.shape)
        print(total_data.mean(axis=(0,1,2))/255)
        print(total_data.std(axis=(0,1,2))/255)

    def sample_pictures(self, train_flag=True, return_flag = False):

        # get some random training images
        dataiter = iter(self.loader(train_flag))
        images,labels = dataiter.next()

        sample_size=25 if train_flag else 5

        images = images[0:sample_size]
        labels = labels[0:sample_size]

        fig = plt.figure(figsize=(10, 10))

        # Show images
        for idx in np.arange(len(labels.numpy())):
                ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
                npimg = unnormalize(images[idx])
                ax.imshow(npimg, cmap='gray')
                ax.set_title("Label={}".format(str(self.classes[labels[idx]])))

        fig.tight_layout()  
        plt.show()

        if return_flag:
            return images,labels

class album_Compose_tiny_imagenet_train():
    def __init__(self):
        self.albumentations_transform = Compose([
            PadIfNeeded(70,70),
            RandomCrop(64,64),
            HorizontalFlip(),
            #RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5),
            #GaussianBlur(blur_limit=3, p=0.25),
            #ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.30, rotate_limit=45, p=.35),
            Cutout(num_holes=1, max_h_size=20, max_w_size=20, fill_value=[0.485*255, 0.456*255, 0.406*255], always_apply=True, p=1.00),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ToTensor()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class album_Compose_tiny_imagenet_test():
    def __init__(self):
        self.albumentations_transform = Compose([
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ToTensor()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class dataset_tiny_imagenet:
    """
    Class to load the data and define the data loader
    """

    def __init__(self, batch_size=128):

        # Defining CUDA
        cuda = torch.cuda.is_available()
        print("CUDA availability ?",cuda)

        # Defining data loaders with setting
        self.dataloaders_args = dict(shuffle=True, batch_size = batch_size, num_workers = 4, pin_memory = True) if cuda else dict(shuffle=True,batch_size = batch_size)
        self.sample_dataloaders_args = self.dataloaders_args.copy()

        self.classes = []
        self.path = './tiny-imagenet-200/'

        id_dict = {}
        for i, line in enumerate(open(self.path + 'wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
            self.classes.append(line.replace('\n', ''))

        self.train_dir = os.path.join(self.path, 'train')
        self.test_dir = os.path.join(self.path, 'val')
  

        all_classes = {}
        result = {}
        for i, line in enumerate(open( self.path + 'words.txt', 'r')):
            n_id, word = line.replace('\n', '').split('\t')[:2]
            all_classes[n_id] = word
        for key, value in id_dict.items():
            result[value] = (key, all_classes[key])      
        self.class_to_id_dict = result

    def data(self, train_flag):

        # Transformations data augmentation (only for training)
        if train_flag :
            return datasets.ImageFolder(self.train_dir,
                                        transform=album_Compose_tiny_imagenet_train())

        # Testing transformation - normalization adder
        else:
            return datasets.ImageFolder(self.test_dir,
                                        transform=album_Compose_tiny_imagenet_test())

    # Dataloader function
    def loader(self, train_flag=True):
        return(torch.utils.data.DataLoader(self.data(train_flag), **self.dataloaders_args))


    def data_summary_stats(self):
        # Load train data as numpy array
        train_data = self.data(train_flag=True)
        test_data = self.data(train_flag=False)

        total_data = np.concatenate((train_data, test_data), axis=0)
        print(total_data.shape)
        print(total_data.mean(axis=(0,1,2))/255)
        print(total_data.std(axis=(0,1,2))/255)

    def sample_pictures(self, train_flag=True, return_flag = False):

        # get some random training images
        dataiter = iter(self.loader(train_flag))
        images,labels = dataiter.next()

        sample_size=25 if train_flag else 5

        images = images[0:sample_size]
        labels = labels[0:sample_size]

        fig = plt.figure(figsize=(10, 10))

        # Show images
        for idx in np.arange(len(labels.numpy())):
                ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
                npimg = unnormalize(images[idx])
                ax.imshow(npimg, cmap='gray')
                ax.set_title("Label={}".format(str(self.classes[labels[idx]])))

        fig.tight_layout()  
        plt.show()

        if return_flag:
            return images,labels

def unnormalize(img):
    channel_means = (0.485, 0.456, 0.406)
    channel_stdevs = (0.229, 0.224, 0.225)
    img = img.numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
         img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
    return np.transpose(img, (1,2,0))