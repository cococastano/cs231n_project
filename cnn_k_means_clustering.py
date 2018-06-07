import numpy as np
import os.path
import data_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as T


# will we be using GPUs?
USE_GPU = False
if USE_GPU and torch.cuda.is_available(): device = torch.device('cuda')
else: device = torch.device('cpu')

# float values used
dtype = torch.float32
# constant to control how often we print training loss
train_batch_size = 32 # 32
val_batch_size = 32 # 5
print_every = int(100/train_batch_size)

# number of dimensions to cluster by
num_dims = 3

# plotting stuff
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

# load data
class_names = ['no break', 'break']  # 0 is no break and 1 is break
frame_range = list(range(0,3))

frame_range = [0, 3, 7]
num_classes = len(class_names)
num_train = 1500*len(frame_range) # 3400
num_val = 100*len(frame_range)# # 200
num_test = 50*len(frame_range) # 188

# make user provide model name to save to avoid overwriting if it exists 
# already! (mostly for me)
print('give model name:')
model_name = input()

model_file = 'C:/Users/nicas/Documents/CS231N-ConvNNImageRecognition/' + \
             'Project/' + model_name + '.pt'

if os.path.isfile(model_file) is False:
    (X_train, y_train,
     X_val, y_val, X_test, y_test) = data_utils.get_data(frame_range=frame_range,
                                                         num_train=num_train,
                                                         num_validation=num_val,
                                                         num_test=num_test,
                                                         feature_list=None,
                                                         reshape_frames=False,
                                                         crop_at_constr=False)
    _, _, im_h, im_w = X_train.shape
    print('train data shape: ', X_train.shape)
    print('train labels shape: ', y_train.shape)
    print('validation data shape: ', X_val.shape)
    print('validation labels shape: ', y_val.shape)
    print('test data shape: ', X_test.shape)
    print('test labels shape: ', y_test.shape)
    print()
    
    # create tesor objects, normalize and zero center and pass into data loaders
    # hardcoded mean and standard deviation of pixel values
    mean_pv, std_pv = 109.23, 99.78  # turns out its not helpful for binary data
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    # create data loader objects
    train = torch.utils.data.TensorDataset(X_train, y_train)
    # arrays are already randomized so shuffle=False
    loader_train = torch.utils.data.DataLoader(train, shuffle=True, 
                                               batch_size=train_batch_size)
    val = torch.utils.data.TensorDataset(X_val, y_val)
    loader_val = torch.utils.data.DataLoader(val, shuffle=True, 
                                             batch_size=val_batch_size)
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
        for X, y in loader:
            # move to device, e.g. GPU or CPU
            X = X.to(device=device, dtype=dtype)  
            y = y.to(device=device, dtype=torch.long)
            scores = model(X)
            # get locations of max in each row
            _, preds = scores.max(1)
            num_correct += (preds == y.squeeze(1)).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        if print_out is not False:
            print('got %d / %d correct (%.2f)' % (num_correct, 
                  num_samples, 100 * acc))
    return acc  

def encode_data(loader, model):
    # set model to evaluation mode
    model.eval()
    dims = []
    classes = []
    index = 0
    with torch.no_grad():
        for X, y in loader:
            # move to device, e.g. GPU or CPU
            X = X.to(device=device, dtype=dtype)  
            y = y.to(device=device, dtype=torch.long)
            dims.append(np.array(model.encode(X,encode_to_n_dims=3))[0])
            classes.append(np.array(y)[0][0])
            index += 1
            if index%500 == 0: print('encoding dataset', index)
    return (dims,classes)

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
        print()
        print('TRAINING EPOCH: ',e)
        for t, (X, y) in enumerate(loader_train):
            # put model in training mode
            model.train() 
            # move to device, e.g. GPU
            X = X.to(device=device, dtype=dtype)  
            y = y.to(device=device, dtype=torch.long)
                        
            scores = model(X)

            loss = F.cross_entropy(scores, y.squeeze(1))

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
  
class OneLayerEncoder(nn.Module):
    # just to make sure our process feeding into k-means is ok
    def __init__(self, in_channel, channel_1, num_dims):
        super().__init__()
        # initialize 2D conv layer 1
        self.c2d_1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, stride=1,
                               padding=2)
        # initialize fully connected layers of 2D conv layers
        self.fc_1_train = nn.Linear(im_h*im_w*channel_1, num_classes)
        self.fc_1_encode = nn.Linear(im_h*im_w*channel_1, 100)
        self.fc_2_encode = nn.Linear(100, num_dims)
        
    def forward(self,x):
        # forward pass conv net
        x = F.relu(self.c2d_1(x))
        x = flatten(x)
        x = self.fc_1_train(x)
        return x
    
    def encode(self,x):
        # forward pass layer to encode
        x = F.relu(self.c2d_1(x))
        x = flatten(x)
        x = self.fc_1_encode(x)
        x = self.fc_2_encode(x)
        return x

class SixLayerEncoder(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, channel_3,
                 channel_4, num_dims):
        super().__init__()
        self.num_dims = num_dims
        # initialize 2D conv layer 1
        self.c2d_1 = nn.Conv2d(in_channel, channel_1, kernel_size=7, stride=1,
                               padding=3)
        # initialize 2D conv layer 2
        self.c2d_2 = nn.Conv2d(channel_1, channel_2, kernel_size=5, stride=1,
                               padding=2)
        # initialize maxpool
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2)
        # initialize 2D conv layer 3
        self.c2d_3 = nn.Conv2d(channel_2, channel_3, kernel_size=3, stride=1,
                               padding=1)
        # initialize 2D conv layer 4
        self.c2d_4 = nn.Conv2d(channel_3, channel_4, kernel_size=3, stride=1,
                               padding=1)
        # initialize maxpool
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2)
        # initialize fully connected layers of 2D conv layers
        self.fc_1 = nn.Linear(channel_4*im_h*im_w/16, 100)
        self.fc_2_train = nn.Linear(100, num_classes)
        self.fc_2_encode = nn.Linear(100, num_dims)
        
    def forward(self,x):
        # forward pass for 2*(conv -> relu -> conv -> relu -> pool) -> fc ->
        # relu -> fc
        x = F.relu(self.c2d_1(x))
        x = F.relu(self.c2d_2(x))
        x = self.maxpool2d_1(x)
        x = F.relu(self.c2d_3(x))
        x = F.relu(self.c2d_4(x))
        x = self.maxpool2d_2(x)
        x = flatten(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2_train(x)
        return x

    def encode(self,x,encode_to_n_dims):
        # forward pass layer to encode
        x = F.relu(self.c2d_1(x))
        x = F.relu(self.c2d_2(x))
        x = self.maxpool2d_1(x)
        x = F.relu(self.c2d_3(x))
        x = F.relu(self.c2d_4(x))
        x = self.maxpool2d_2(x)
        x = flatten(x)
        x = self.fc_1(x)
        x = F.relu(x)
        if encode_to_n_dims is self.num_dims:
            x = self.fc_2_encode(x)
        else:
            fc_last = nn.Linear(100, encode_to_n_dims)
            x = fc_last(x)
        return x
    

################################## SCRIPT #####################################

if os.path.isfile(model_file) is False:  # train and save model
    print('training new model...')
    ##### intialize parameters of 1-layer ConvNet
#    learning_rate = 1e-3
#    channel_1 = 16 
#    
#    model_1 = OneLayerEncoder(in_channel=1, channel_1=channel_1, 
#                              num_dims=num_dims)
#    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#    optimizer = optim.Adam(model_1.parameters(), lr=learning_rate)
#    loss_history, val_acc_history, train_acc_history = \
#        train_model(model_1, optimizer, epochs=10, return_history=True)
#            
#    check_accuracy(loader_test, model_1, training=False, print_out=True)
        
    #### intialize parameters of 6 layer ConvNet
    learning_rate = 1e-3 # 3e-4 gave max of almost 97% on val, 
    channel_1, channel_2, channel_3, channel_4= 32, 16, 8, 4
    model_2 =  SixLayerEncoder(in_channel=1, channel_1=channel_1,
                               channel_2=channel_2, channel_3=channel_3,
                               channel_4=channel_4, num_dims=num_dims)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model_2.parameters(), lr=learning_rate)
    loss_history, val_acc_history, train_acc_history = \
        train_model(model_2, optimizer, epochs=2, return_history=True)
        
    check_accuracy(loader_test, model_2, training=False, print_out=True)
        
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
    
    # save the model if its good!
    torch.save(model_2, model_file)
    
else:
    
#    model_name = 'six_layer_encoder_v1'
#    model_file = 'C:/Users/nicas/Documents/CS231N-ConvNNImageRecognition/' + \
#             'Project/' + model_name + '.pt'
    
    # to avoid memory problems partition data calls and loop over frame ranges
    def chunks(l, n):
        """yield successive n-sized chunks from l"""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    print('loading existing model...')
    my_model = torch.load(model_file)
    encoded_vals = []
    
    class_names = ['no break', 'break']  # 0 is no break and 1 is break
    frame_range = list(range(0,51))
    num_classes = len(class_names)
    part_frame_range = list(chunks(frame_range,2))
    for i, sub_range in enumerate(part_frame_range):
        print()
        print('PULLING PARTITION ', i)
        num_train = 3400*len(sub_range) # 3400
        num_val = 200*len(sub_range) # 200
        num_test = 188*len(sub_range) # 188
        (X_train, y_train,
         X_val, y_val, X_test, y_test) = \
             data_utils.get_data(frame_range=sub_range,
                                 num_train=num_train,
                                 num_validation=num_val,
                                 num_test=num_test,
                                 feature_list=None,
                                 reshape_frames=False,
                                 crop_at_constr=False)
                     
        # create tesor objects, normalize and zero center and pass into data 
        #loaders
        # hardcoded mean and standard deviation of pixel values
        #data
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
    
        
        X_all = torch.cat((X_train,X_val,X_test),0)
        y_all = torch.cat((y_train,y_val,y_test),0)
        all_data = torch.utils.data.TensorDataset(X_all, y_all)
        print()
        print('data shape: ', X_all.shape)
        loader_all_data = torch.utils.data.DataLoader(all_data, shuffle=True)
        # pull out encoded dims
        dims,classes = encode_data(loader_all_data,my_model)
        
        fig3D = plt.figure(1)  # LEO vs angle vs area
        ax = fig3D.add_subplot(111, projection='3d')
        for i,dim in enumerate(dims):
            if int(classes[i]) is 1: color = 'r'
            if int(classes[i]) is 0: color = 'b'
            x = dim[0]
            y = dim[1]
            z = dim[2]
            encoded_vals.append((x,y,z,classes[i]))
            ax.scatter(x, y, z, c=color)
        
        ax.set_xlabel('dimension 1')
        ax.set_ylabel('dimension 2')
        ax.set_ylabel('dimension 3')
    
