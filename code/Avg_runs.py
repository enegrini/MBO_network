"""Code for Threshold Dynamics on MNIST"""
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import os
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import numpy as np
from PIL import Image
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




import argparse

def main():
    """Function to traun MBO network, can also compute average across multiple runs"""
    parser = argparse.ArgumentParser(description='Average training results')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs to average')
    parser.add_argument('--steep_list', type=float, nargs='+', default=np.linspace(100,300, 3), help='List of steepness values')
    parser.add_argument('--n_epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--multilayer', type=bool, default=True, help='Use all frames in training')
    parser.add_argument('--kernel_size', type=int, default=31, help='Kernel size')
    parser.add_argument('--data_path', type=str, default='K31standard_T7dT1.5_Noise_None', help='Data path')
    parser.add_argument('--device', type=str, default="cuda:0" , help='device for training')

    args = parser.parse_args()

    average_train(args.runs, args.steep_list, args.n_epochs, args.lr, args.batch_size, args.multilayer, int(args.kernel_size), args.data_path, args.device)

# Create a single class for the convolutional network
class FlexibleConvNet(nn.Module):
    def __init__(self, num_layers, kernel, threshold,activation,steepness = 1e2, device='cuda:0'):
        super(FlexibleConvNet, self).__init__()
        self.num_layers = num_layers
        self.shared_kernel = nn.Parameter(torch.Tensor(kernel).unsqueeze(0).unsqueeze(0))
        self.shared_threshold = torch.Tensor([threshold]).to(device)#nn.Parameter(torch.Tensor([threshold]))
        self.activation = activation
        self.steepness = steepness

    def forward(self, x):
        layer_outputs = []  # List to store the output of each layer
        for _ in range(self.num_layers):
            # Apply 2D convolution with 'same' padding using the shared kernel
            x = nn.functional.conv2d(x, self.shared_kernel, padding='same')
            # Apply a custom activation function with the shared threshold
            x = self.activation(x,self.shared_threshold,self.steepness)
            layer_outputs.append(x)  # Append the output of the current layer to the list
        return layer_outputs  # Return the list of layer outputs

#define activation
#for datageneration heaviside is fine, but we cannot use it in training.
def smooth_step_function(x, threshold, steepness=1e2):
    return 1 / (1 + torch.exp(-steepness * (x - threshold)))

#define loss
def multilayer_loss(outputs,targets):
    tot_loss = 0.
    for i in range(targets.shape[2]):
        mse_loss = nn.functional.mse_loss(outputs[i], targets[:,:,i,:,:])
        tot_loss += mse_loss
    return(tot_loss)

def train(n_epochs=500, lr=1e-4,batch_size = 20, multilayer=True, steepness = 1e2, kernel_size = 15, data_path = 'K31hard_T7dT1.5_Noise_None',device = 'cuda:0'):
    print('device in train', device)
    input_images = np.load(os.path.join(data_path, 'TrainInput.npy'))#np.load(os.path.join(data_path, 'TrainInput_'+data_path+'.npy'))
    target_outputs = np.load(os.path.join(data_path, 'TrainTarget.npy'))#np.load(os.path.join(data_path, 'TrainTarget_'+data_path+'.npy'))
    shared_kernel = np.random.rand(kernel_size,kernel_size)
    shared_kernel= shared_kernel/np.sum(shared_kernel)
    shared_threshold = 0.25
    time = 3#target_outputs.shape[1] #this is the number of frames
    net = FlexibleConvNet(num_layers=time, kernel=shared_kernel, threshold=shared_threshold,activation = smooth_step_function,steepness=steepness, device= device).to(device)
    if multilayer:
        print('Training with all frames')
        # Define the loss function (Mean Squared Error)
        criterion = multilayer_loss

    else:
        print('Training with only the last frame')
        # Define the loss function (Mean Squared Error)
        criterion = nn.MSELoss()

    # Create a DataLoader for the generated dataset
    dataset = torch.utils.data.TensorDataset(torch.tensor(input_images, dtype=torch.float), torch.tensor(target_outputs[:,:time,:,:], dtype=torch.float))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the optimizer (Stochastic Gradient Descent)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Define your learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=500, factor=1/2, verbose=True)

    
    # Training loop
    net.train()
    loss_epoch = []
    for epoch in range(n_epochs):
        total_loss = 0.0
        for inputs, targets in data_loader:
            # Move the inputs and targets to the selected device
            inputs, targets = inputs.unsqueeze(1).to(device), targets.unsqueeze(1).to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = net(inputs)
            # Calculate the loss
            if multilayer:
                loss = criterion(output, targets)
            else:
                loss = criterion(output[-1], targets[:,:,-1,:,:])


            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        scheduler.step(total_loss / len(data_loader))
        loss_epoch.append(total_loss / len(data_loader))
        if epoch%100==0:
            print(f"Epoch [{epoch+1}/{n_epochs}] - Loss: {loss_epoch[-1]:.4f}")
        if math.isnan(loss_epoch[-1]):
            print("NaN loss encountered. Stopping training.")
            break

    ## Compute scale:
    kernel_learn = net.shared_kernel.data.squeeze().cpu().detach().numpy()
    scaling_factor = kernel_learn.sum()
    kernel_learn = kernel_learn/scaling_factor
    threshold_learn = net.shared_threshold.data.item()/scaling_factor
    
    return kernel_learn, threshold_learn, loss_epoch

def average_train(runs=2, steep_list=np.linspace(100,300, 3), n_epochs=300, lr=1e-4, batch_size=1000, multilayer=True, kernel_size=31, data_path='K31hard_T7dT1.5_Noise_None', device = 'cuda:0'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print('device = ', device)
    
    kernel_means = []
    threshold_means = []
    
    nan_loss_encountered = False

    for steepness in steep_list:
        print("steepness", steepness)
        kernels = []
        thresholds = []

        for i in range(runs):
            print('run = ', i)
            kernel_learn, threshold_learn, loss_epoch = train(n_epochs, lr, batch_size, multilayer, steepness, kernel_size, data_path,device)

            # Check if loss is NaN before appending to the list
            if not math.isnan(loss_epoch[-1]):
                kernels.append(kernel_learn)
                thresholds.append(threshold_learn)
            else:
                print('NaN loss, no kernels learned')
                nan_loss_encountered = True
                break  # Exit the inner loop

        if not nan_loss_encountered:
            kernel_means.append(np.mean(np.array(kernels), axis=0))
            threshold_means.append(np.mean(np.array(thresholds), axis=0))
        else:
            print('NaN loss, no mean computed')
            break  # Exit the outer loop
    
    print('Saving Kernels and Thresholds')
    # Create a subfolder called "learned_KT" in the data_path
    subfolder_path = os.path.join(data_path, 'learned_KT')
    os.makedirs(subfolder_path, exist_ok=True)

    # Save kernels and thresholds inside the subfolder
    np.save(os.path.join(subfolder_path, 'kernels_' + str(kernel_learn.shape[0]) + '.npy'), np.array(kernel_means))
    np.save(os.path.join(subfolder_path, 'thresholds_' + str(kernel_learn.shape[0]) + '.npy'), np.array(threshold_means))
    return kernel_means, threshold_means


if __name__ == '__main__':
    main()






