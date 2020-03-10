# Torch libraries
import torch                   #PyTorch base libraries
from torchvision import datasets, transforms


class dataset_cifar10:
    """
    Class to load the data and define the data loader
    """

    def __init__(self, sample=True, batch_size=128):

        # Defining CUDA
        cuda = torch.cuda.is_available()
        print("CUDA availability ?",cuda)

        # batch_size=128
        # mean_tuple=(0.4914, 0.4822, 0.4465)
        # std_tuple =(0.2023, 0.1994, 0.2010)

        # if sample:
        #     batch_size=5 
        #     mean_tuple=()
        #     std_tuple =()

        self.sample = sample

        # Defining data loaders with setting
        self.dataloaders_args = dict(shuffle=True, batch_size=batch_size, num_workers = 4, pin_memory = True) if cuda else dict(shuffle=True,batch_size=64)

    #     # Transformations in training phase
    #     self.train_aug_transforms_list=[
    #                                     transforms.RandomCrop(32, padding=4),
    # #                                    transforms.RandomHorizontalFlip(),
    # #                                    transforms.RandomRotation(10),
    #                                     transforms.RandomHorizontalFlip(),
    #                                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    #                                    ]
    #     # Transformations in testing phase
    #     self.test_transforms_list = [
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize(mean_tuple, std_tuple),
    #                                 ]




        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def data(self, train_flag):

        if self.sample:
            # Transformations in testing phase
            trnsfm = transforms.Compose([
            #                                      transforms.RandomCrop(32, padding=4),
            #                                      transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()
                                                ])
        elif train_flag:
            # Transformations in training phase
            trnsfm = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
        #                                       transforms.RandomHorizontalFlip(),
        #                                       transforms.RandomRotation(10),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])
        else:
        # Transformations in testing phase
            trnsfm = transforms.Compose([
            #                                      transforms.RandomCrop(32, padding=4),
            #                                      transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                 ])

        # trnsfm = transforms.Compose(
        #     self.train_aug_transforms_list.append(self.test_transforms_list) if train_flag else self.test_transforms_list
        # )


        # Loading data
        return datasets.CIFAR10('./Data',
                                train=train_flag,
                                transform=trnsfm,
                                download=True)

    def loader(self, train_flag=True):
        return(torch.utils.data.DataLoader(self.data(train_flag), **self.dataloaders_args))
