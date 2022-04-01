import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

'''Figures'''

figure_path = '/content/drive/MyDrive/QMUL/NN/figures/' 
def plot_acc(train_acc, val_acc, epochs, savefig=False, name='name'):
  plt.plot(np.arange(1,epochs+1), [100*i for i in train_acc], label='training')
  plt.plot(np.arange(1,epochs+1), [100*i for i in val_acc], label='validation')
  plt.legend()
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  if savefig:
      plt.savefig(figure_path+name)    

def plot_cfmat(model, data, dict_classes, savefig=False, name='name'):
    y_all = []
    pred_all = []
    with torch.no_grad():
        for X, y in data:
            model.eval()
            outputs = model(X)
            _, pred = torch.max(outputs, 1)
            y_all.append(y)
            pred_all.append(pred)
    y_cat = torch.cat([y for y in y_all])
    pred_cat = torch.cat([pred for pred in pred_all])
    cm = confusion_matrix(y_cat.cpu().numpy(), pred_cat.cpu().numpy(), labels=[range(7)])
    count = np.sum(cm, axis=1).reshape(-1,1)
    cm = cm*100 / count 
    cm = pd.DataFrame(data=cm, index=list(dict_classes.values()), 
                              columns=list(dict_classes.values()))

    plt.figure(figsize=(7,7))
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues')
    plt.xlabel("Prediction", fontsize=13, rotation=0)
    plt.ylabel("True labels", fontsize=13)
    if savefig:
      plt.savefig(figure_path+name)    

'''Training'''

def train_loop(dataloader, model, loss_fn, optimizer, print_loss=False):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if print_loss:
            if (batch+1) % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    return correct

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    # print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

# def fit(model, loss, optimizer, epochs, train_data, val_data, print_loss=False):
#     list_train_acc = []
#     list_val_acc = []
#     for t in range(epochs):
#         print(f"Epoch {t+1}")
#         train_acc = train_loop(train_data, model, loss, optimizer, print_loss=print_loss)
#         val_acc = test_loop(val_data, model, loss)
#         print(f"Train accuracy: {(100*train_acc):>0.1f}% \nVal accuracy  : {(100*val_acc):>0.1f}%")
#         list_train_acc.append(train_acc)
#         list_val_acc.append(val_acc)
#         print('-------------------------------')
#     print("Done!")
#     return list_train_acc, list_val_acc

def fit(
    model, 
    optimizer, 
    epochs, 
    train_data, 
    val_data, 
    train_func=train_loop,
    test_func=test_loop,
    train_loss=nn.CrossEntropyLoss(),
    test_loss=nn.CrossEntropyLoss(),
    print_loss=False):

    list_train_acc = []
    list_val_acc = []
    for t in range(epochs):
        print(f"Epoch {t+1}")
        train_acc = train_func(train_data, model, train_loss, optimizer, print_loss=print_loss)
        val_acc = test_func(val_data, model, test_loss)
        print(f"Train accuracy: {(100*train_acc):>0.1f}% \nVal accuracy  : {(100*val_acc):>0.1f}%")
        list_train_acc.append(train_acc)
        list_val_acc.append(val_acc)
        print('-------------------------------')
    print("Done!")
    return list_train_acc, list_val_acc


'''label smoothing'''
def linear_combination(x, y, epsilon):
    return (1 - epsilon) * x + epsilon * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(nll, loss/n, self.epsilon)

'''mixup'''
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_train_loop(
  dataloader, 
  model, 
  criterion, 
  optimizer, 
  train_loss = 0,
  correct = 0,
  # total = 0,
  print_loss=False):
    size = len(dataloader.dataset)
    model.train()
    for batch, (inputs, targets) in enumerate(dataloader):
        # Compute prediction and loss
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
        # inputs, targets_a, targets_b = map(Variable, (inputs,
        #                                               targets_a, targets_b))
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        # total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if print_loss:
            if (batch+1) % 100 == 0:
                loss, current = loss.item(), batch * len(inputs)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    return correct

# def mixup_fit(model, loss, optimizer, epochs, train_data, val_data, print_loss=False):
#     list_train_acc = []
#     list_val_acc = []
#     for t in range(epochs):
#         print(f"Epoch {t+1}")
#         train_acc = mixup_train_loop(
#           train_data, 
#           model, 
#           loss, 
#           optimizer, 
#           print_loss=print_loss)
#         val_acc = test_loop(val_data, model, loss)
#         print(f"Train accuracy: {(100*train_acc):>0.1f}% \nVal accuracy  : {(100*val_acc):>0.1f}%")
#         list_train_acc.append(train_acc)
#         list_val_acc.append(val_acc)
#         print('-------------------------------')
#     print("Done!")
#     return list_train_acc, list_val_acc

# def mixup_ls_fit(model, train_loss, test_loss, optimizer, epochs, train_data, val_data, print_loss=False):
#     list_train_acc = []
#     list_val_acc = []
#     for t in range(epochs):
#         print(f"Epoch {t+1}")
#         train_acc = mixup_train_loop(
#           train_data, 
#           model, 
#           train_loss, 
#           optimizer, 
#           print_loss=print_loss)
#         val_acc = test_loop(val_data, model, test_loss)
#         print(f"Train accuracy: {(100*train_acc):>0.1f}% \nVal accuracy  : {(100*val_acc):>0.1f}%")
#         list_train_acc.append(train_acc)
#         list_val_acc.append(val_acc)
#         print('-------------------------------')
#     print("Done!")
#     return list_train_acc, list_val_acc

'''random seed'''
def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms = True

'''weighted CrossEntropyLoss'''
def make_weight(dl, dev):
    label = np.array([i for _, i in dl.dataset])
    label_weight = np.array([np.sum(label==i) for i in range(7)])
    label_weight = 1/label_weight
    label_weight = label_weight/np.sum(label_weight)
    return torch.from_numpy(label_weight.astype(np.float32)).clone().to(dev)
