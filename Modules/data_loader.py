# Torch libraries
import torch                   #PyTorch base libraries
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, Cutout
from albumentations.pytorch import ToTensor

class album_Compose_train():
    def __init__(self):
        self.albumentations_transform = Compose([
            RandomCrop(30,30),
            HorizontalFlip(),
            Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=[0.4914, 0.4822, 0.4465], always_apply=False, p=0.5),
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

        fig = plt.figure(figsize=(10, 5))

        # Show images
        for idx in np.arange(len(labels.numpy())):
                ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
                npimg = np.transpose(images[idx].numpy(),(1,2,0))
                ax.imshow(npimg, cmap='gray')
                ax.set_title("Label={}".format(str(self.classes[labels[idx]])))

        fig.tight_layout()  
        plt.show()

        if return_flag:
            return images,labels