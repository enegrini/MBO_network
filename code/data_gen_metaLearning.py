"""Code for generating data for Threshold Dynamics on MNIST"""
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
import scipy.ndimage
import os
import cv2
import random
import ast

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from utils import *

# Set the device (CPU or GPU) for data generation cpu is fine
device = "cpu"
data_folder = 'video_data/data_Final'
os.makedirs(data_folder, exist_ok=True)
kernel_folder = 'video_data/true_kernels_Final'
os.makedirs(kernel_folder, exist_ok=True)
threshold_folder = 'video_data/true_thresholds_Final'
os.makedirs(threshold_folder, exist_ok=True)
images_folder = 'video_data/video_examples_Final'
os.makedirs(images_folder, exist_ok=True)

def main():
    """Function to generate data for metalearning MBO network"""
    parser = argparse.ArgumentParser(description='Generate training and validation data')
    parser.add_argument('--time', type=int, default=7, help='Frames in video')
    parser.add_argument('--thresholds', type=str, default='0.2,0.5', help='Threshold values, comma-separated list')
    parser.add_argument('--deltat', type=float, default=1, help='Step between each frame')
    parser.add_argument('--kernel_dim', type=int, default=31, help='Kernel dimension')
    parser.add_argument('--img_size', type=int, default=100, help='Image size')
    parser.add_argument('--dataset_size', type=int, default=20, help='Dataset size')
    parser.add_argument('--kernel_kinds', type=str, default="['standard']", help='Kernel type (standard, hard, hard2, circle, MNIST)')
    parser.add_argument('--noise_kinds', type=str, default="['None']", help='Noise type (None, blur, or SP)')
    parser.add_argument('--runs', type=int, default=2, help='How many runs of data generation')

    args = parser.parse_args()
    kernel_kinds = ast.literal_eval(args.kernel_kinds)
    noise_kinds = ast.literal_eval(args.noise_kinds)
    thresholds = [float(x) for x in args.thresholds.split(',')]
    for run in range(args.runs):
        for kernel_kind in kernel_kinds:
            for threshold in thresholds:
                for noise_kind in noise_kinds:
                    generate_data(args.time, threshold, args.deltat, args.kernel_dim, args.img_size, args.dataset_size,
                      kernel_kind, noise_kind,run)

# Create a single class for the convolutional network
class FlexibleConvNet(nn.Module):
    def __init__(self, num_layers, kernel, threshold,activation,steepness = 1e2):
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

def generate_data(time, threshold, deltat, kernel_dim, img_size, dataset_size, kernel_kind, noise_kind, run):
    """Function to generate data for meta learning MBO"""
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True)


    # Define the FlexibleConvNet with "time" number of layers
    image_size = img_size #set desired image size
    kernel_dim = kernel_dim  # Dimension of the square Gaussian kernel
    
    if kernel_kind == 'standard':
        np.random.seed(run)
        mean = np.random.randint(kernel_dim//4, kernel_dim-kernel_dim//4) # Mean of the Gaussian kernel
        std = np.random.randint(kernel_dim//2, kernel_dim)        # Standard deviation of the Gaussian kernel
        kernel = Gaussian_Kernel(kernel_dim, mean, std, deltat)
        kernel_kind = kernel_kind + 'M' + str(mean) + "S" + str(std)
    elif kernel_kind == 'hard':
        np.random.seed(run)
        rotation_angle = np.random.randint(10,90)
        kernel1 = Gaussian_Kernel_double(kernel_dim, 15,15, image_size/2 ,image_size/5 , deltat,rotation_angle)
        kernel2 = Gaussian_Kernel_double(kernel_dim, 15,15, image_size/2 ,image_size/5 , deltat,rotation_angle)
        kernel = (kernel1+kernel2)/2
        kernel_kind = kernel_kind + 'A' + str(rotation_angle)
    
    elif kernel_kind == 'hard2':
        np.random.seed(run)
        if threshold > 0.4:
            deltat = 0.2   # Delta t between two successive frames, this is multiplied by std
        mean1 =  np.random.randint(kernel_dim//4, kernel_dim-kernel_dim//4) # Mean of the Gaussian kernel
        mean2 =  np.random.randint(kernel_dim//4, kernel_dim-kernel_dim//4) 
        std = np.random.randint(kernel_dim//2, kernel_dim)        # Standard deviation of the Gaussian kernel
        kernel1 = Gaussian_Kernel_double(kernel_dim, mean1, mean1, std,std, deltat,rotation_angle=0)
        kernel2 = Gaussian_Kernel_double(kernel_dim,mean2,mean2, std,std, deltat,rotation_angle=0)
        kernel = (kernel1+kernel2)/2
        kernel_kind = kernel_kind + 'M1' + str(mean1) + "M2" + str(mean2) + "S" + str(std)
    
    elif kernel_kind == 'circle':
        np.random.seed(run)
        radius=np.random.randint(kernel_dim//6, kernel_dim//4)
        center_x=np.random.randint(-kernel_dim//10, kernel_dim//10)
        center_y=np.random.randint(-kernel_dim//10, kernel_dim//10)
        kernel = generate_circle_img(radius, center_x, center_y, resolution=kernel_dim)
        kernel = kernel/np.sum(kernel)
        kernel_kind = kernel_kind + str(radius) + 'C' + str(center_x)+str(center_y)

    elif kernel_kind == 'MNIST':
        np.random.seed(run)
        deltat = 0.2
        ker_index = np.random.randint(len(mnist_dataset))
        kern_img, label = mnist_dataset[ker_index]
        kern_img = np.squeeze(np.array(kern_img))
        min_val = np.min(kern_img)
        max_val = np.max(kern_img)
        kern_img = (kern_img - min_val) / (max_val - min_val)
        digit_sz = kernel_dim-10
        # Resize the digit to be smaller (e.g., 20x20)
        smaller_img = resize(kern_img, (digit_sz, digit_sz))
        # Create a 31x31 frame
        kernel = np.zeros((31, 31))
        # Center the smaller image in the frame
        start_x = (31 - digit_sz) // 2
        start_y = (31 - digit_sz) // 2
        kernel[start_y:start_y+digit_sz, start_x:start_x+digit_sz] = smaller_img
        # Ensure the sum of all pixel values is 1
        kernel = kernel/np.sum(kernel)
        # Apply Gaussian blur to the image
        kernel = cv2.GaussianBlur(kernel, (5, 5), 0)
        kernel_kind = kernel_kind + str(label) + 'id' + str(ker_index)
        
    # Create the "data" folder if it doesn't exist
    data_path = 'K'+ str(kernel_dim) +str(kernel_kind) +'_T' + str(time) + '_thr' +str(threshold) + '_Noise_'+ str(noise_kind)
    plt.figure(figsize=(6,6))
    plt.imshow(kernel)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('True Kernel')
    plt.tight_layout()
    plt.savefig(os.path.join(kernel_folder, 'Kernel_'+ str(data_path)+'.png'))
    np.save(os.path.join(kernel_folder, 'kernel_true'+ str(data_path)+'.npy'), np.array(kernel))
    plt.close()
    
    
    file_path = threshold_folder + '/threshold_values' + str(data_path) + '.txt'
    # Open the file in append mode and write the threshold value
    with open(file_path, 'w') as file:
        file.write(f'{threshold}\n')
        
    net = FlexibleConvNet(num_layers=time, kernel=kernel, threshold=threshold, activation = heaviside).to(device)

    # Create a dataset with modified images and target outputs of all layers
    target_outputs = []
    for _ in range(dataset_size):
        # Randomly select an MNIST image from the dataset
        img_index = np.random.randint(len(mnist_dataset))
        mnist_image, _ = mnist_dataset[img_index]
        mnist_image = resize(np.squeeze(np.array(mnist_image)), (image_size,image_size))

        # Modify the MNIST image
        modified_image = np.array(np.where(mnist_image[0] > 0, 1, 0), dtype=np.float32)#np.where(mnist_image[0] > 0, mnist_image[0], 0) #np.where(mnist_image[0] > 0, 1, 0)


        # Convert the modified image to a PyTorch tensor
        modified_image_tensor = torch.tensor(modified_image).unsqueeze(0).float()

        # Pass the modified image through the network to get the outputs of all layers
        layer_outputs = net(modified_image_tensor)
        layer_outputs_tensor = np.squeeze(torch.stack(layer_outputs, dim=1).detach().numpy())
    #     print(layer_outputs_tensor.shape)

        # Append the modified image and all layer outputs to the dataset
        target_outputs.append(layer_outputs_tensor)

    # Convert the lists to NumPy arrays
    target_outputs_clean = np.array(target_outputs)
    
    if noise_kind == 'None':
        target_outputs = target_outputs_clean
        
    elif noise_kind == 'blur':
        target_outputs_blur = np.zeros_like(target_outputs_clean)
        for img_idx in range(len(target_outputs_clean)):
            for frame in range(target_outputs_clean.shape[1]):
                target_outputs_blur[img_idx,frame,:,:] = apply_blur(target_outputs_clean[img_idx,frame,:,:])
        target_outputs = target_outputs_blur
        
    elif noise_kind == 'SP':
        target_outputs_SP = np.zeros_like(target_outputs_clean)
        for img_idx in range(len(target_outputs_clean)):
            for frame in range(target_outputs_clean.shape[1]):
                target_outputs_SP[img_idx,frame,:,:] = apply_salt_and_pepper(target_outputs_clean[img_idx,frame,:,:])
        target_outputs = target_outputs_SP
    
    # Plot target data
    random_integers = random.sample(range(dataset_size), 3)
    num_rows = len(random_integers)
    num_cols = time  # Only frames
    
    plt.figure(figsize=(10, 6))
    for idx, i in enumerate(random_integers):
        for j in range(time):
            plt.subplot(num_rows, num_cols, idx * num_cols + j + 1)
            plt.imshow(target_outputs[i, j, :, :])
            plt.title(f'frame {j + 1}')
    
    # Save the figure into the "data" folder
    plt.tight_layout()
    plt.savefig(os.path.join(images_folder, f'Data_Noise_{data_path}.png'))
    plt.close()

    
    #save data
    train_target_name = str(data_path)
    print(train_target_name)
    
    # Save the NumPy arrays into the "data" folder
    train_target_path = os.path.join(data_folder, f'{train_target_name}.npy')
   

    np.save(train_target_path, target_outputs)
    
    return 




if __name__ == '__main__':
    main()
