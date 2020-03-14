# Torch libraries
import torch                   #PyTorch base libraries
from torchvision import datasets, transforms


class dataset_cifar10:
    """
    Class to load the data and define the data loader
    """

    def __init__(self, sample_batch_size=5, batch_size=128):

        # Defining CUDA
        cuda = torch.cuda.is_available()
        print("CUDA availability ?",cuda)

        # Defining data loaders with setting
        args = dict(shuffle=True, num_workers = 4, pin_memory = True) if cuda else dict(shuffle=True)
        self.dataloaders_args = args.update({'batch_size':batch_size})
        self.sample_dataloaders_args = args.update({'batch_size':sample_batch_size})

        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def data(self, train_flag , trnsfm_flag=True):

        trnsfm_list = []

        if train_flag:
            # Transformations data augmentation (only for training)
            trnsfm_list.extend([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                               ])

        # Data base transform
        trnsfm_list.append(transforms.ToTensor())

        if trnsfm_flag:
            # Testing transformation - normalization adder
            trnsfm_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        trnsfm = transforms.Compose(trnsfm_list)

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
        return(torch.utils.data.DataLoader(self.data(train_flag), **self.sample_dataloaders_args))
