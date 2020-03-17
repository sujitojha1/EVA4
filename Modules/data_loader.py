# Torch libraries
import torch                   #PyTorch base libraries
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

!pip install -U git+https://github.com/albu/albumentations
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, Cutout
from albumentations.pytorch import ToTensor

class dataset_cifar10:
    """
    Class to load the data and define the data loader
    """

    def __init__(self, sample_batch_size=5, batch_size=128):

        # Defining CUDA
        cuda = torch.cuda.is_available()
        print("CUDA availability ?",cuda)

        # Defining data loaders with setting
        self.dataloaders_args = dict(shuffle=True, batch_size = batch_size, num_workers = 4, pin_memory = True) if cuda else dict(shuffle=True,batch_size = batch_size)
        self.sample_dataloaders_args = self.dataloaders_args.copy()
        self.sample_dataloaders_args['batch_size'] = sample_batch_size

        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    def data(self, train_flag , trnsfm_flag=True):

        trnsfm_list = [ToTensor()]

        if trnsfm_flag:

            # Transformations data augmentation (only for training)
            if train_flag :
                aug_list = [
                            RandomCrop(32, padding=4),
                            HorizontalFlip(),
                            Cutout(max_holes=8, max_height=16, max_width=16, fill_value=[0.4914, 0.4822, 0.4465]),
                            ]
                trnsfm_list = aug_list + trnsfm_list

            # Testing transformation - normalization adder
            trnsfm_list.append(Normalize(mean=[0.4914, 0.4822, 0.4465],std=[.2023, 0.1994, 0.2010]))

        trnsfm = Compose(trnsfm_list)
        # Loading data
        return datasets.CIFAR10('./Data',
                                train=train_flag,
                                transform=trnsfm,
                                download=True)

    # Dataloader function
    def loader(self, train_flag=True):
        return(torch.utils.data.DataLoader(self.data(train_flag), **self.dataloaders_args))

    # Sample Dataloader Function
    def sample_loader(self, train_flag=True):
        return(torch.utils.data.DataLoader(self.data(train_flag,trnsfm_flag=False), **self.sample_dataloaders_args))

    def data_summary_stats(self):
        # Load train data as numpy array
        train_data = self.data(train_flag=True,trnsfm_flag=False).data
        test_data = self.data(train_flag=False,trnsfm_flag=False).data

        total_data = np.concatenate((train_data, test_data), axis=0)
        print(total_data.shape)
        print(total_data.mean(axis=(0,1,2))/255)
        print(total_data.std(axis=(0,1,2))/255)

    def sample_pictures(self, train_flag=True , return_flag = False):

        # get some random training images
        dataiter = iter(self.sample_loader(train_flag))
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