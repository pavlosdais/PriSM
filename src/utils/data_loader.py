import torch
from torchvision import datasets, transforms

def import_dataset(dataset_n, seed=0):
    classes_n = 10
    torch.manual_seed(seed)

    # MNIST dataset
    if dataset_n == "mnist":
        in_channels = 1
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),]))

        test_dataset = datasets.MNIST('../data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),]))
    
    # cifar10 dataset
    elif dataset_n == "cifar10":
        in_channels = 3

        mean = (0.4914, 0.4822, 0.4465) 
        std = (0.2471, 0.2435, 0.2616)

        train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)
                                 ]))
    
        # Test dataset
        test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)
                                        ]))

    else: return None

    return classes_n, in_channels, train_dataset, test_dataset
