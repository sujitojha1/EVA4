# Torch libraries
import torch                   #PyTorch base libraries
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


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
            trnsfm = transforms.Compose([
                                         transforms.Pad(4,padding_mode='edge'),
                                         transforms.RandomCrop(32),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                         transforms.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(0.25, 0.25), value=[0.4914*255, 0.4822*255, 0.4465*255]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                         ])
            return datasets.CIFAR10('./Data',
                            train=train_flag,
                            transform=trnsfm,
                            download=True)

        # Testing transformation - normalization adder
        else:
            trnsfm = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                         ])
            return datasets.CIFAR10('./Data',
                                train=train_flag,
                                transform=trnsfm,
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


def unnormalize(img):
    channel_means = (0.4914, 0.4822, 0.4465)
    channel_stdevs = (.2023, 0.1994, 0.2010)
    img = img.numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
         img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
    return np.transpose(img, (1,2,0))