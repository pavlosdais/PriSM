# train given target model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from mnist_models.models import Surrogate_MNIST, Model_A, Model_B, Model_C
from models.train import Train
from utils.data_loader import import_dataset

if __name__ == '__main__':

    dataset_n = "mnist"

    # hyperparameters
    batch_s    = 128
    epochs     = 20
    learning_r = 0.01

    # get data
    classes_n, in_channels, train_data, test_data = import_dataset(dataset_n)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_s, shuffle=True, num_workers=8)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_s, shuffle=True, num_workers=8)

    model = Surrogate_MNIST()

    # define model
    optimizer = optim.SGD(model.parameters(), lr=learning_r, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.3)

    use_gpu = False
    if use_gpu:
        device = 'cuda'
        model.cuda()
    else: device = 'cpu'

    # train model and save its weights
    trainer = Train(model, device, epochs, optimizer, train_loader, test_loader, criterion, scheduler)
    trainer.train_nn()
    
    trainer.save_model("surrogate_model")
