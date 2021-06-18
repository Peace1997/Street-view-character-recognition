import json,glob,torch
import numpy as np

from torch.utils.data.dataset import Dataset
from DataSet_Loader import DataSet
from DataSet_Loader import train_loader
from DataSet_Loader import val_loader
from DataSet_Loader import test_loader

from Model import Model_CNN
from Model import Model_Resnet
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import os


if __name__ == '__main__':

    best_loss = 1000.0
    epochs = 2000
    #is_predicting = False #默认为False，为训练过程
    device = torch.device("cpu")

    model = Model_Resnet().to(device)

    loss_func = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    # if (os.path.exists('./resnet_model.pt')):
    #     model.load_state_dict(torch.load('./resnet_model.pt'))
    for epoch in range(epochs):

        print('Epoch:', epoch)

        train_loss,train_accuracy = model.mytrain(train_loader,loss_func, optimizer,device)
        val_loss,val_accuracy = model.myvalidat(val_loader,loss_func,device)
        writer = SummaryWriter('path/to/log')
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('val_accuracy', val_accuracy, epoch)
        #print(model.predict(test_loader,device))
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        print("train_loss:",train_loss,"val_loss:",val_loss)
        if val_loss < best_loss:
            best_epoch,best_loss = epoch,val_loss
            torch.save(model.state_dict(), './resnet_model.pt')
