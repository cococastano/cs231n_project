import torch
import torchvision
import matplotlib.pyplot as plt
import data_utils
import numpy as np
import torch.nn.functional as F
import os
import cv2
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

############################## CLASSES AND METHODS ############################
def flatten(x):
    # read in N, C, H, W
    N = x.shape[0]
    # flatten the the C * H * W images into a single vector per image
    return x.view(N, -1)

def chunks(l, n):
    """yield successive n-sized chunks from l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]
            
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 176, 124)
    return x

def check_accuracy(loader, model, training=False, print_out=False):
    if training is True:
        print('checking accuracy on validation set')
    else:
        print('checking accuracy on test set')   
    total_loss = 0
    num_samples = 0
    # set model to evaluation mode
    model.eval()
    criterion = nn.KLDivLoss()
    with torch.no_grad():
        for img, _ in loader:
            # move to device, e.g. GPU or CPU
            img = img.to(device=device, dtype=dtype)  
            out = model(img)
            loss = criterion(out, img)
            total_loss += loss
            num_samples += 1
        mean_loss = total_loss / num_samples
        if print_out is not False:
            print('mean loss: %.2f' % (mean_loss))
    return mean_loss  


def train_model(model, optimizer, epochs=1, sub=1, return_history=False):
    """
    inputs:
    - model: a PyTorch Module giving the model to train.
    - optimizer: an Optimizer object to train the model
    - epochs: (optional) integer giving the number of epochs to train for
    - return_history: will return tuple of loss, train accuracy, and 
                      validation accuracy histories
    - sub: just to keeps track of the data partition sub range
    
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
    # loss criterion
    criterion = nn.MSELoss()        
    for e in range(epochs):
        print()
        print('TRAINING EPOCH: ',e)
        for t, (img, _) in enumerate(train_loader):
            # put model in training mode
            model.train() 
            # move to device, e.g. GPU
            img = img.to(device=device, dtype=dtype)
            out = model(img)
            loss = criterion(out, img)
            # zero gradients for the variables which the optimizer will update
            optimizer.zero_grad()
            # backward pass: compute the gradient of the loss with respect to 
            # each  parameter of the model
            loss.backward()
            # update the parameters of the model using the gradients computed 
            # by the backwards pass
            optimizer.step()
            if t % 10 == 0:
                print('iteration %d of %d, loss = %.4f' % (t, 
                                                           len(train_loader),
                                                           loss.item()))
                acc = check_accuracy(val_loader, model, 
                                     training=True, print_out=True)
                print()
            
            if return_history is True: loss_history.append(loss)
        
        if return_history is True: 
            val_acc_history.append(acc)
            train_acc_history.append(check_accuracy(val_loader, model, 
                                                    training=True, 
                                                    print_out=False))
        if e % 10 == 0:
            pic = to_img(out.cpu().data)
            orig_pic = to_img(img.cpu().data)
            print('saving images')
            save_image(pic,'./deconv_frames/' +
                       'decode_image_sub{}_epoch{}.png'.format(sub,e))
            save_image(orig_pic,'./deconv_frames/' +
                       'orig_image_sub{}_epoch{}.png'.format(sub,e))
    
    return (loss_history, train_acc_history, val_acc_history)
  

class autoencoder(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2,n_dims=3):
        super(autoencoder, self).__init__()
        # encode
        self.conv_2d_1 = nn.Conv2d(in_channel, channel_1, kernel_size=7, 
                                   stride=1, padding=3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2d_2 = nn.Conv2d(channel_1, channel_2,kernel_size=3, 
                                   stride=1, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # decode   
        self.conv_trans_2d_1 = nn.ConvTranspose2d(channel_2, channel_1, 
                                                  kernel_size=2, stride=1)
        self.conv_trans_2d_2 = nn.ConvTranspose2d(channel_1, channel_2, 
                                                  kernel_size=2, stride=2) 
                                                  #padding=1)
        self.conv_trans_2d_2_v1 = nn.ConvTranspose2d(channel_1, in_channel, 
                                                     kernel_size=2, stride=2) 
                                                  #padding=1)
        self.conv_trans_2d_3 = nn.ConvTranspose2d(channel_2, in_channel, 
                                                  kernel_size=2, stride=2)
                                                  #padding=1)
                                                  
        # encode to n dims
        self.num_dims = num_dims
        # self.conv_to_dims_2d_1 = 
        # self.maxpool_to dims_2 = 
        self.fc_1_encode = nn.Linear(87*61*channel_2,500)
        self.fc_2_encode = nn.Linear(500,100)
        self.fc_3_encode = nn.Linear(100,n_dims)
        
        
    def decode(self, x):
        x = F.relu(self.conv_trans_2d_1(x))
        x = F.tanh(self.conv_trans_2d_2_v1(x))
        return x
    
    def encode(self, x):
        x = F.relu(self.conv_2d_1(x))
        x = self.maxpool_1(x)
        x = F.relu(self.conv_2d_2(x))
        x = self.maxpool_2(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def get_encoded_im(self,x,im_i=0):
        x = self.encode(x)
        x = x.detach().numpy()
        out = x[im_i,0,:,:]
        return out
    
    def get_decoded_im(self,x,im_i=0):
        x = self.decode(x)
        x = x.detach().numpy()
        out = x[im_i,0,:,:]
        return out
    
    def encode_to_n_dims(self,x,n):
        x = self.encode(x)
        x = flatten(x)
        x = self.fc_1_encode(x)
        x = self.fc_2_encode(x)
        if n is self.num_dims:
            x = self.fc_3_encode(x)
        else:
            fc_last = nn.Linear(100, n)
            x = fc_last(x)
        return x

################################## SCRIPT #####################################

# make user provide model name to save to avoid overwriting if it exists 
# already! (mostly for me)
print('give model name:')
model_name = 'auto_encode_v1'#input()
model_file = 'C:/Users/nicas/Documents/CS231N-ConvNNImageRecognition/' + \
             'Project/' + model_name + '.pt'

# plotting stuff
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

class_names = ['no break', 'break']  # 0 is no break and 1 is break
# we will partition the frame range randomly to train encoder

if os.path.isfile(model_file) is False:
    
    if not os.path.exists('./deconv_frames'):
        os.mkdir('./deconv_frames')
    
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
    in_channel = 1
    channel_1 = 16
    channel_2 = 8
    learning_rate = 1e-3
    num_epochs = 50
    model = autoencoder(in_channel=in_channel, channel_1=channel_1,
                        channel_2=channel_2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)

    frame_range = list(range(0,50))
    part_num = 2  # number of partitions
    frame_range = np.random.permutation(np.array(frame_range))
    part_frame_range = list(chunks(frame_range,part_num))
    num_classes = len(class_names)
    num_train = 3000*part_num # 3400
    num_val = 200*part_num # 200
    num_test = 188*part_num # 188
    
    # store all history
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # we want to train only paritions of data because loading all fills memory
    for i, sub_range in enumerate(part_frame_range):   
    
        (X_train, y_train,
         X_val, y_val, 
         X_test, y_test) = data_utils.get_data(frame_range=sub_range,
                                               num_train=num_train,
                                               num_validation=num_val,
                                               num_test=num_test,
                                               feature_list=None,
                                               reshape_frames=False,
                                               crop_at_constr=True,
                                               blur_im=True)
             
        _, _, im_h, im_w = X_train.shape
        
        # create tesor objects, normalize and zero center and pass into data 
        # loaders hardcoded mean and standard deviation of pixel values
        mean_pv, std_pv = 109.23, 99.78  # turns out its not helpful for binary data
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
        # collect all
        X_all = torch.cat((X_train,X_test),0)
        y_all = torch.cat((y_train,y_test),0)
        print('train data shape: ', X_train.shape)
        print('train label shape: ', y_train.shape)
        print('val data shape: ', X_val.shape)
        print('val label shape: ', y_val.shape)
        print()
        all_data = torch.utils.data.TensorDataset(X_all, y_all)
        val_data = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(all_data, shuffle=True, 
                                                  batch_size=train_batch_size)
        val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, 
                                                  batch_size=val_batch_size)

        print('TRAINING SUB RANGE %d OF %d' %(i, len(part_frame_range)))
        sub_loss_history, sub_val_acc_history, sub_train_acc_history = \
            train_model(model, optimizer, epochs=num_epochs, sub=i, 
                        return_history=True)
            
        loss_history.append(sub_loss_history)
        val_acc_history.append(sub_val_acc_history)
        train_acc_history.append(sub_train_acc_history)
        
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
    torch.save(model, model_file)
        
