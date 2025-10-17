import torch
from torchvision import datasets, transforms

def import_dataset(dataset_n, seed=0):
    torch.manual_seed(seed)

    # MNIST dataset
    if dataset_n == "mnist":
        classes_n = 10
        in_channels = 1
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                       ]))
        test_dataset = datasets.MNIST('../data', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    
    # CIFAR-10 dataset
    elif dataset_n == "cifar10":
        classes_n = 10
        in_channels = 3
        mean = (0.4914, 0.4822, 0.4465) 
        std = (0.2471, 0.2435, 0.2616)
        train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)
                                         ]))
        test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)
                                        ]))

    # ImageNet Mini Dataset (using Kaggle path)
    elif dataset_n == "imagenet":
        classes_n   = 1000
        in_channels = 3
        
        imagenet_path = '/kaggle/input/imagenetmini-1000/imagenet-mini/'

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        try:
            train_dataset = datasets.ImageFolder(root=imagenet_path + 'train', transform=transform)
            test_dataset = datasets.ImageFolder(root=imagenet_path + 'val', transform=transform)
        except FileNotFoundError:
            print(f"\n[ERROR] ImageNet dataset not found at the specified path: {imagenet_path}")
            return None, None, None, None

    else:
        raise ValueError(f"Unknown dataset: {dataset_n}")

    return classes_n, in_channels, train_dataset, test_dataset