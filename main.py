import random
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torch.backends.cudnn
from tqdm import tqdm
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
color_mean = [0.485, 0.456, 0.406]
color_std = [0.229, 0.224, 0.225]
BATCH_SIZE = 50
EPOCHS = 80
MODEL_NAME = "model"

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
parameters = {'axes.labelsize': 25,
              'axes.titlesize': 35}
plt.rcParams.update(parameters)


def evaluate_model(model, test_loader, loss_function):
    model.eval()
    model.to(device)
    torch.cuda.empty_cache()
    total_loss, correct, total = 0, 0, 0

    for images, labels in test_loader :
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        total_loss += loss_function(output, labels).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(test_loader)
    return average_loss, correct, total

def load_cifar10_data(batch_size):
    train_augmentation = T.Compose([
        T.Pad(4),
        T.RandomHorizontalFlip(),
        T.RandomCrop(32),
        T.ToTensor(),
        T.Normalize(color_mean, color_std)
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(color_mean, color_std)
    ])
    train_dataset = datasets.CIFAR10(root='./cifar_10data/',
                                     train=True,
                                     transform=train_augmentation,
                                     download=True)
    test_dataset = datasets.CIFAR10(root='./cifar_10data/',
                                    train=False,
                                    transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def train(
    model : nn.Module, epochs : int,
    train_loader : DataLoader, validation_loader,
    loss_func,
    optimizer,
    scheduler,
    save_path,
    plot
):
    model.train()
    torch.cuda.empty_cache()
    start = time.time()
    loss_values = []
    accr_values = []
    loss_valids = []
    accr_valids = []
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = 0
        print(f"EPOCH {epoch+1} training begins...")
        correct, total = 0, 0
        for i, data in enumerate(tqdm(train_loader)):
            img, mask = data
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)
            loss = loss_func(output, mask)
            train_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(mask.view_as(pred)).sum().item()
            total += mask.size(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = train_loss/len(train_loader)
        epoch_accr = 100*correct/total
        loss_values.append(epoch_loss)
        accr_values.append(epoch_accr)
        print(f"Train epoch {epoch+1} / {epochs}",
              f"Loss {epoch_loss:.4f}, Accuracy {epoch_accr:.2f}%",
              f"Training Time {(time.time()-epoch_start)/60:.2f} min")
        if validation_loader is not None:
            model.eval()
            valid_loss = 0
            valid_correct, valid_total = 0, 0
            with torch.no_grad():
                for i, data in enumerate(tqdm(validation_loader)):
                    img, mask = data
                    img = img.to(device)
                    mask = mask.to(device)
                    output = model(img)
                    valid_loss += loss_func(output, mask).item()
                    pred = output.max(1, keepdim=True)[1]
                    valid_correct += pred.eq(mask.view_as(pred)).sum().item()
                    valid_total += mask.size(0)
            valid_loss = valid_loss/len(validation_loader)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()
            valid_accr = 100*valid_correct/valid_total
            loss_valids.append(valid_loss)
            accr_valids.append(valid_accr)
            if max(accr_valids) == valid_accr and save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f"saved model to {save_path}")
            print(f"Validation epoch {epoch+1} / {epochs}",
                  f"Loss {valid_loss:.4f}, Accuracy {valid_accr:.2f}%")
        print()
    print(f"Total training time {(time.time()-start)/60:.2f} minutes taken")
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.plot(loss_values)
        ax1.plot(loss_valids)
        ax1.set_title("Loss")
        ax2.plot(accr_values)
        ax2.plot(accr_valids)
        ax2.set_title("Accuracy (%)")

class Solver:
    def __init__(self, model, epochs, loss_func, optimizer, scheduler):
        self.model = model
        self.epochs = epochs
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler

def solve_cifar10(configuration):
    global MODEL_NAME
    train_loader, test_loader = load_cifar10_data(BATCH_SIZE)
    model = configuration.model.to(device)
    MODEL_NAME = model.name
    optimizer = configuration.optimizer
    scheduler = configuration.scheduler
    loss_function = configuration.loss_function
    print(summary(model, (3, 32, 32)))
    print(f"DATASET : CIFAR-10\nMODEL : {MODEL_NAME}-{EPOCHS}\nOPTIMIZER : {optimizer}\nSCHEDULER : {scheduler}")
    train (
        model = model,
        epochs = EPOCHS,
        train_loader = train_loader,
        validation_loader = test_loader,
        loss_func = loss_function,
        optimizer = optimizer,
        scheduler = scheduler,
        save_path = f"./pretrained/{MODEL_NAME}-{EPOCHS}",
        plot=True
    )
    average_loss, correct, total = evaluate_model(model, test_loader, loss_function)
    print(f"DATASET : CIFAR-10\nMODEL : {MODEL_NAME}-{EPOCHS}\nOPTIMIZER : {optimizer}\nSCHEDULER : {scheduler}\n"
          f"TESTSET AVG LOSS : {average_loss:.2f}\n"
          f"TESTSET AVG ACCR : {100*correct/total:.2f}%")
    plt.savefig(f"./result_graphs/{MODEL_NAME}.png")

class Config:
    def __init__(self, model, optimizer, scheduler, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function

# 66.77%
def cifar10_LeNet():
    model = LeNet()
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = 2000
    EPOCHS = 50
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.95)
    config_LeNet = Config(
        model = model,
        optimizer = optimizer,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.2),
        loss_function = torch.nn.CrossEntropyLoss()
    )
    solve_cifar10(config_LeNet)

# 85.03% (76 min training)
def cifar10_AlexNet():
    model = AlexNet()
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = 500
    EPOCHS = 50
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
    config_AlexNet = Config(
        model = model,
        optimizer = optimizer,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.1),
        loss_function = torch.nn.CrossEntropyLoss()
    )
    solve_cifar10(config_AlexNet)

# 83.16% (31 min training)
def cifar10_SmallAlexNet():
    model = SmallAlexNet()
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = 500
    EPOCHS = 50
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
    config_AlexNet = Config(
        model = model,
        optimizer = optimizer,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.1),
        loss_function = torch.nn.CrossEntropyLoss()
    )
    solve_cifar10(config_AlexNet)

# vgg11 : 88.79% (17 min training)
# vgg13 : 90.40% (24 min training)
# vgg16 : 90.73% (31 min training)
# vgg19 : 90.01% (38 min training)
def cifar10_VGGNet(vggnet_option):
    model = VGGNet(vggnet_option)
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = 100
    EPOCHS = 50
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.0005, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
    config = Config(
        model = model,
        optimizer = optimizer,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor = 0.5),
        loss_function = torch.nn.CrossEntropyLoss()
    )
    solve_cifar10(config)

# 89.94% (204 min training)
def cifar10_GoogLeNet():
    model = GoogLeNet()
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = 50
    EPOCHS = 50
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
    config_GoogLeNet = Config(
        model = model,
        optimizer = optimizer,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5),
        loss_function = torch.nn.CrossEntropyLoss()
    )
    solve_cifar10(config_GoogLeNet)

def cifar10_ResNet(resnet_option):
    model = ResNet(resnet_option)
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = 500
    EPOCHS = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)
    config = Config(
        model = model,
        optimizer = optimizer,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5),
        loss_function = torch.nn.CrossEntropyLoss()
    )
    solve_cifar10(config)

def cifar10_LargeResNet(resnet_option):
    model = LargeResNet(resnet_option)
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = 50
    EPOCHS = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)
    config = Config(
        model = model,
        optimizer = optimizer,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5),
        loss_function = torch.nn.CrossEntropyLoss()
    )
    solve_cifar10(config)

#cifar10_GoogLeNet()
#cifar10_LeNet()
#cifar10_AlexNet()
#cifar10_SmallAlexNet()
RESNET_OPTIONS = ['resnet20-cifar', 'resnet32-cifar', 'resnet44-cifar', 'resnet56-cifar']
for option in RESNET_OPTIONS:
    cifar10_ResNet(option)

VGGNET_OPTIONS = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
#for option in VGGNET_OPTIONS:
    #cifar10_VGGNet(option)
#cifar10_GoogLeNet()
#cifar10_LargeResNet('resnet-orig101')