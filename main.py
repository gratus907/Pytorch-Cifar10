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

    average_loss = total_loss / total
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
    model : nn.Module,
    epochs : int,
    train_loader : DataLoader,
    validation_loader,
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
            if min(accr_valids) == valid_accr and save_path is not None:
                torch.save(model.state_dict(), save_path)
            print(f"Validation epoch {epoch+1} / {epochs}",
                  f"Loss {valid_loss:.4f}, Accuracy {valid_accr:.2f}%")

    print(f"Total training time {(time.time()-start)/60:.2f} minutes taken")
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.plot(loss_values)
        ax1.plot(loss_valids)
        ax1.set_title("Training loss")
        ax2.plot(accr_values)
        ax2.plot(accr_valids)
        ax2.set_title("Training accuracy (%)")

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

def cifar10_GoogLeNet():
    model = GoogLeNet()
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = 500
    EPOCHS = 50
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
    config_GoogLeNet = Config(
        model = model,
        optimizer = optimizer,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5),
        loss_function = torch.nn.CrossEntropyLoss()
    )
    solve_cifar10(config_GoogLeNet)

def cifar10_LeNet():
    model = LeNetRegularized()
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = 2000
    EPOCHS = 200
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.95)
    config_LeNet = Config(
        model = model,
        optimizer = optimizer,
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.96),
        loss_function = torch.nn.CrossEntropyLoss()
    )
    solve_cifar10(config_LeNet)

#cifar10_GoogLeNet()
cifar10_LeNet()
