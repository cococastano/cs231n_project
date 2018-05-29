# -*- coding: utf-8 -*-
"""
Created on Tue May 29 01:26:16 2018

@author: nicas
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 29 00:40:09 2018

@author: nicas
"""

import numpy as np
import extract_features
import data_utils
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F


# will we be using GPUs?
USE_GPU = False
if USE_GPU and torch.cuda.is_available(): device = torch.device('cuda')
else: device = torch.device('cpu')

# float values used
dtype = torch.float32
# constant to control how often we pint training loss
print_every = 100

# plotting stuff
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

# load data
num_train = 1000
num_val = 200
num_test = 184
(X_train, y_train, 
 X_val, y_val, X_test, y_test) = data_utils.get_data(num_train=num_train,
                                                     num_validation=num_val,
                                                     num_test=num_test,
                                                     feature_list=None,
                                                     reshape_frames=False)
_, _, im_h, im_w = X_train.shape
print('train data shape: ', X_train.shape)
print('train labels shape: ', y_train.shape)
print('validation data shape: ', X_val.shape)
print('validation labels shape: ', y_val.shape)
print('test data shape: ', X_test.shape)
print('test labels shape: ', y_test.shape)
print()

# create tesor objects to pass into data loaders 
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)
# create data loader objects
train = torch.utils.data.TensorDataset(X_train, y_train)
# arrays are already randomized so shuffle=False
loader_train = torch.utils.data.DataLoader(train, shuffle=True)
val = torch.utils.data.TensorDataset(X_val, y_val)
loader_val = torch.utils.data.DataLoader(val, shuffle=True)
test = torch.utils.data.TensorDataset(X_test, y_test)
loader_test = torch.utils.data.DataLoader(test, shuffle=True)

######################## USEFUL METHODS AND CLASSES ##########################
def flatten(x):
    # read in N, C, H, W
    N = x.shape[0]
    # flatten the the C * H * W images into a single vector per image
    return x.view(N, -1)

def check_accuracy(loader, model, training=False, print_out=False):
    if training is True:
        print('checking accuracy on validation set')
    else:
        print('checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            # move to device, e.g. GPU or CPU
            x = x.to(device=device, dtype=dtype)  
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            # get locations of max in each row
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        if print_out is not False:
            print('got %d / %d correct (%.2f)' % (num_correct, 
                  num_samples, 100 * acc))
    return acc

def train_model(model, optimizer, epochs=1, return_history=False):
    """
    inputs:
    - model: a PyTorch Module giving the model to train.
    - optimizer: an Optimizer object to train the model
    - epochs: (optional) integer giving the number of epochs to train for
    - return_history: will return tuple of loss, train accuracy, and 
                      validation accuracy histories
    
    returns: nothing, but prints model accuracies during training.
    """
    # move the model parameters to CPU/GPU
    model = model.to(device=device)
    if return_history is not False: 
        loss_history = []
        train_acc_history = []
        val_acc_history = []
    else:
        loss_history = None
        train_acc_history = None
        val_acc_history = None
    for e in range(epochs):
        print('TRAINING EPOCH: ',e)
        for t, (x, y) in enumerate(loader_train):
            # put model in training mode
            model.train() 
            # move to device, e.g. GPU
            x = x.to(device=device, dtype=dtype)  
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            
            loss = F.cross_entropy(scores, y[0])

            # zero gradients for the variables which the optimizer will update
            optimizer.zero_grad()

            # backward pass: compute the gradient of the loss with respect to 
            # each  parameter of the model
            loss.backward()

            # update the parameters of the model using the gradients computed 
            # by the backwards pass
            optimizer.step()

            if t % print_every == 0:
                print('iteration %d, loss = %.4f' % (t, loss.item()))
                acc = check_accuracy(loader_val, model, 
                                     training=True, print_out=True)
                print()
            
            if return_history is True: loss_history.append(loss)
        
        if return_history is True: 
            val_acc_history.append(acc)
            train_acc_history.append(check_accuracy(loader_val, model, 
                                                    training=True, 
                                                    print_out=False))
    
    return (loss_history, train_acc_history, val_acc_history)

class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        # initialize 2D conv layer 1
        self.c2d_1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, stride=1,
                               padding=2)
        # initialize 2D conv layer 2
        self.c2d_2= nn.Conv2d(channel_1, channel_2, kernel_size=3, stride=1,
                               padding=1)
        # initialize fully connected layers of 2D conv layers
        self.cd2_fc1 = nn.Linear(im_h*im_w*channel_2, num_classes)
        
    def forward(self,x):
        # forward pass for three layer conv net
        x = F.relu(self.c2d_1(x))
        x = F.relu(self.c2d_2(x))
        x = flatten(x)
        scores = self.cd2_fc1(x)
        return scores
    
class ThreeLayerConvNetBeta(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        # initialize 2D conv layer 1
        self.c2d_1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, stride=1,
                               padding=2)
        # initialize 2D conv layer 2
        self.c2d_2= nn.Conv2d(channel_1, channel_2, kernel_size=3, stride=1,
                               padding=1)
        # initialize fully connected layers of 2D conv layers
        self.cd2_fc1 = nn.Linear(im_h*im_w*channel_2, 1)
        self.cd2_fc2 = nn.Linear(1, num_classes)
        
    def forward(self,x):
        # forward pass for three layer conv net
        x = F.relu(self.c2d_1(x))
        x = F.relu(self.c2d_2(x))
        print('after relu of conv2')
        print(x.shape)
        x = F.max_pool2d(x, 2)
        print('after MAX pool')
        print(x.shape)
        x = flatten(x)
        print('after flatten')
        print(x.shape)
        x = self.cd2_fc1(x)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        scores = self.cd2_fc2(x)
        print(scores.shape)
        print('scores above')
        return scores
    
class FiveLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, channel_3,
                 channel_4, num_classes):
        super().__init__()
        # initialize 2D conv layer 1
        self.c2d_1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, stride=1,
                               padding=2)
        # initialize 2D conv layer 2
        self.c2d_2= nn.Conv2d(channel_1, channel_2, kernel_size=3, stride=1,
                               padding=1)
        # initialize fully connected layers of 2D conv layers
        self.cd2_fc1 = nn.Linear(im_h*im_w*channel_2, num_classes)
        
    def forward(self,x):
        # forward pass for three layer conv net
        x = F.relu(self.c2d_1(x))
        x = F.max_pool2d(x)
        x = F.relu(self.c2d_2(x))
        x = flatten(x)
        scores = self.cd2_fc1(x)
        return scores
    

#def test_ThreeLayerConvNet():
#    x = torch.zeros((23, 1, 176, 288), dtype=dtype)
#    model = ThreeLayerConvNet(in_channel=1, channel_1=32, channel_2=16, 
#                              num_classes=10)
#    scores = model(x)
#    print(scores.size())
#    return True
#test_ThreeLayerConvNet()







# intialize parameters of 3-layer ConvNet
learning_rate = 5e-6
channel_1, channel_2 = 32, 16

model = ThreeLayerConvNet(in_channel=1, channel_1=channel_1,
                          channel_2=channel_2, num_classes=2)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_history, val_acc_history, train_acc_history = \
    train_model(model, optimizer, epochs=4, return_history=True)
    
plt.subplot(2,1,1)
plt.plot(loss_history, '-o')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.subplot(2,1,2)
plt.plot(train_acc_history, '-o')
plt.plot(val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


check_accuracy(loader_test, model, print_out=True)

#conv_w1 = random_weight((channel_1,3,5,5))
#conv_b1 = zero_weight((channel_1))
#conv_w2 = random_weight((channel_2,channel_1,3,3))
#conv_b2 = zero_weight((channel_2))
#fc_w = random_weight((channel_2*32*32,10))
#fc_b = zero_weight((10))
#
#
#
#
#
#X_train = np.reshape(X_train,(num_train,176,288))
#print(X_train.shape)
#
#
#plt.figure()
#plt.imshow(X_train[1,:,:])#.astype('uint8'))
#
#plt.figure()
#plt.imshow(X_train[4,:,:])#.astype('uint8'))
#
#
#classes = ['no break', 'break']
#num_classes = len(classes)
#samples_per_class = 6
#
#subplot_fig, axs = plt.subplots(samples_per_class, 2,figsize=(15,12))
#subplot_fig.subplots_adjust(hspace=0.5,wspace=0.2)
#
## normalize 
#
#for y, cls in enumerate(classes):
#    idxs = np.flatnonzero(y_train == y)
#    idxs = np.random.choice(idxs, samples_per_class, replace=False)
#    for i, idx in enumerate(idxs):
#        plt_idx = i * num_classes + y + 1
#        print(idx)
#        plt.subplot(samples_per_class, num_classes, plt_idx)
#        plt.imshow(X_train[idx,:,:])#.astype('uint8'))
#        plt.axis('off')
#        if i == 0:
#            plt.title(cls)
#plt.show()
#

# try even break no break dist, different optimizer
