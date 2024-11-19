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

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from scipy.ndimage import gaussian_filter

# Set the device (CPU or GPU) for data generation cpu is fine
device = "cpu"
print('device = ', device)

def main():
    """Function to generate data for MBO network"""
    parser = argparse.ArgumentParser(description='Generate training and validation data')
    parser.add_argument('--time', type=int, default=7, help='Frames in video')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold value')
    parser.add_argument('--deltat', type=float, default=1.5, help='Step between each frame')
    parser.add_argument('--kernel_dim', type=int, default=31, help='Kernel dimension')
    parser.add_argument('--img_size', type=int, default=100, help='Image size')
    parser.add_argument('--dataset_size', type=int, default=100, help='Dataset size')
    parser.add_argument('--kernel_kind', type=str, default='standard', help='Kernel type (standard, hard or hard2)')
    parser.add_argument('--mean_x1', type=float, default=15, help='Mean x1 for hard kernel') #these params are useless now
    parser.add_argument('--mean_y1', type=float, default=15, help='Mean y1 for hard kernel')
    parser.add_argument('--mean_x2', type=float, default=15, help='Mean x2 for hard kernel')
    parser.add_argument('--mean_y2', type=float, default=15, help='Mean y2 for hard kernel')
    parser.add_argument('--std_x', type=float, default=None, help='Standard deviation x for hard kernel')
    parser.add_argument('--std_y', type=float, default=None, help='Standard deviation y for hard kernel')
    parser.add_argument('--noise_kind', type=str, default=None, help='Noise type (None, blur, or SP)')

    args = parser.parse_args()
    
    if args.std_x is None:
        args.std_x = args.img_size / 10
    if args.std_y is None:
        args.std_y = args.img_size / 40
    generate_data(args.time, args.threshold, args.deltat, args.kernel_dim, args.img_size, args.dataset_size,
                  args.kernel_kind, args.mean_x1, args.mean_y1, args.mean_x2, args.mean_y2, args.std_x, args.std_y,
                  args.noise_kind)


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

# Generate Gaussian Kernel given mean and stdv
def Gaussian_Kernel(kernel_dim, mean, std,deltat=1):
    """
    Args:
        kernel_dim (int): Dimension of the square Gaussian kernel.
        mean (float): Mean of the Gaussian kernel.
        std (float): Standard deviation of the Gaussian kernel.

    Returns:
        ndarray: Kernel
    """
    kernel = np.exp(-((np.arange(kernel_dim) - mean) ** 2) / (2 * std*deltat ** 2))
    kernel /= np.sum(kernel)
    Ker = kernel[np.newaxis, :] * kernel[:, np.newaxis]
    return Ker

def Gaussian_Kernel_double(kernel_dim, mean_x, mean_y, std_x, std_y, deltat=1, rotation_angle=0):
    """
    Args:
        kernel_dim (int): Dimension of the square Gaussian kernel.
        mean (float): Mean of the Gaussian kernel.
        std_x (float): Standard deviation along the x-axis of the Gaussian kernel.
        std_y (float): Standard deviation along the y-axis of the Gaussian kernel.
        deltat (float): Delta t, optional parameter.
        rotation_angle (float): Rotation angle in degrees.

    Returns:
        ndarray: Kernel
    """
    x = np.arange(kernel_dim) - mean_x
    y = np.arange(kernel_dim) - mean_y

    # Generate 1D Gaussian kernel along the x and y axes
    kernel_x = np.exp(-(x ** 2) / (2 * std_x*deltat**2))
    kernel_y = np.exp(-(y ** 2) / (2 * std_y*deltat**2))

    # Normalize the kernels
    kernel_x /= np.sum(kernel_x)
    kernel_y /= np.sum(kernel_y)

    # Create a 2D Gaussian kernel by multiplying the 1D kernels
    Ker = np.outer(kernel_x, kernel_y)

    # Rotate the kernel
    rotated_kernel = scipy.ndimage.rotate(Ker, rotation_angle, reshape=False)

    return rotated_kernel



def generate_circle_indicator(x, y, radius, center_x, center_y):
    """
    Generates the indicator function of a circle shape.

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        radius (float): radius of the circle.
        center_x (float): x-coordinate of the center of the circle.
        center_y (float): y-coordinate of the center of the circle.

    Returns:
        int: 1 if the point (x, y) is inside the circle, 0 otherwise.
    """
    dist_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    if dist_to_center <= radius:
        return 1
    else:
        return 0


def generate_circle_img(radius, center_x, center_y, resolution=100):
    """
    Plots the indicator function of a circle shape as an image.

    Args:
        radius (float): radius of the circle.
        center_x (float): x-coordinate of the center of the circle.
        center_y (float): y-coordinate of the center of the circle.
        resolution (int): number of points along each axis to generate the image.
                          Higher values result in a smoother image. (default: 100)
    """
    x = np.linspace(-10, 10, resolution)
    y = np.linspace(-10, 10, resolution)
    X, Y = np.meshgrid(x, y)
    print(X.shape)

    Z = np.zeros_like(X, dtype=int)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = generate_circle_indicator(X[i, j], Y[i, j], radius, center_x, center_y)
    return Z.astype(float)


#define activation
#for datageneration heaviside is fine, but we cannot use it in training.

def heaviside(x, threshold=0.0,steepness = None):
    """
    Heaviside step function with a specified threshold.

    Args:
        x (torch.Tensor): Input tensor.
        threshold (float): Threshold value.

    Returns:
        torch.Tensor: Heaviside step function with the specified threshold.
    """
    return torch.where(x > threshold, torch.tensor(1.0, dtype=x.dtype), torch.tensor(0.0, dtype=x.dtype))

#noise functions
def apply_blur(image):
    sigma = 1.0  # Standard deviation for Gaussian kernel
    blurred_image = gaussian_filter(image, sigma=sigma)
    # # Convert NumPy array to PIL Image
    # image_pil = Image.fromarray(image.astype(np.uint8))  # Convert to uint8

    # # Apply Gaussian blur
    # image_blurred_pil = gaussian_blur(image_pil, kernel_size=11)

    return blurred_image

def apply_salt_and_pepper(image, p=0.01):
    # Apply salt-and-pepper noise
    noise = (np.random.rand(*image.shape) < p).astype(np.float32)
    pepper = (np.random.rand(*image.shape) < 0.5).astype(np.float32)
    salt = (np.random.rand(*image.shape) < 0.5).astype(np.float32)
    noisy_image = image * (1 - noise) + pepper * (1 - salt)

    return noisy_image

#resize MNIST images to be of desired dimension. Default is upscale to 100x100
def resize(img, target_size):
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(target_size)
    resized_img = np.array(pil_img)
    return resized_img[np.newaxis, :]

def generate_data(time, threshold, deltat, kernel_dim, img_size, dataset_size, kernel_kind, mean_x1,mean_y1, mean_x2,mean_y2,std_x, std_y, noise_kind):
    """Function to generate training and validation data
    input time = frames in video, deltat = step between each frame, kernel_dim = kernel dimension, image size and datasetsize
    saves input and target images for training"""
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True)


    # Define the FlexibleConvNet with "time" number of layers
    image_size = img_size #set desired image size
    kernel_dim = kernel_dim  # Dimension of the square Gaussian kernel
    
    if kernel_kind == 'standard':
        np.random.seed(42)
        mean =  np.random.randint(kernel_dim//4, kernel_dim-kernel_dim//4) # Mean of the Gaussian kernel
        std = np.random.randint(kernel_dim//10, kernel_dim//2)        # Standard deviation of the Gaussian kernel
        deltat = deltat   # Delta t between two successive frames, this is multiplied by std
        kernel = Gaussian_Kernel(kernel_dim, mean, std, deltat)
        kernel_kind = kernel_kind + 'M' + str(mean) + "S" + str(std)
    elif kernel_kind == 'hard':
        np.random.seed(42)
        deltat = deltat   # Delta t between two successive frames, this is multiplied by std
        kernel1 = Gaussian_Kernel_double(kernel_dim, 15,15, image_size/10 ,image_size/40 , deltat,rotation_angle=30)
        kernel2 = Gaussian_Kernel_double(kernel_dim, 15,15, image_size/10 ,image_size/40 , deltat,rotation_angle=30)
        kernel = (kernel1+kernel2)/2
    elif kernel_kind == 'hard2':
        np.random.seed(42)
        deltat = deltat   # Delta t between two successive frames, this is multiplied by std
        kernel1 = Gaussian_Kernel_double(kernel_dim, 10, 10, image_size/20,image_size/20, deltat,rotation_angle=0)
        kernel2 = Gaussian_Kernel_double(kernel_dim, 20,20, image_size/20,image_size/20, deltat,rotation_angle=0)
        kernel = (kernel1+kernel2)/2
    elif kernel_kind == 'circle':
        np.random.seed(42)
        radius=np.random.randint(kernel_dim//12, kernel_dim//6)
        center_x=np.random.randint(-kernel_dim//10, kernel_dim//10)
        center_y=np.random.randint(-kernel_dim//10, kernel_dim//10)
        kernel = generate_circle_img(radius, center_x, center_y, resolution=kernel_dim)
        kernel_kind = kernel_kind + str(radius) + 'C' + str(center_x)+str(center_y)
        kernel /= np.sum(kernel)
    elif kernel_kind == 'MNIST':
        np.random.seed(42)
        # Select a random image
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
        kernel /= np.sum(kernel)
        # Apply Gaussian blur to the image
        kernel = cv2.GaussianBlur(kernel, (5, 5), 0)
        kernel_kind = kernel_kind + str(label) + 'id' + str(ker_index)
        
    print('kernel', kernel_kind)
    # Create the "data" folder if it doesn't exist
    folder_path = 'K'+ str(kernel_dim) +str(kernel_kind) +'_T' + str(time) +'dT' + str(deltat) + '_thr' +str(threshold) + '_Noise_'+ str(noise_kind)
    os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.imshow(kernel)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('True Kernel')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'Kernel_'+ str(kernel_kind)+'.png'))
    np.save(os.path.join(folder_path, 'kernel_true.npy'), np.array(kernel))
    
    threshold = threshold
    print("Threshold", threshold)
    time = time      # This is the number of frames
    net = FlexibleConvNet(num_layers=time, kernel=kernel, threshold=threshold, activation = heaviside).to(device)

    # Create a dataset with modified images and target outputs of all layers
    train_size = dataset_size
    validation_size = 10#train_size//10
    fulldata_size = train_size + validation_size
    input_images = []
    target_outputs = []
    np.random.seed(24)
    print('generating images')
    for _ in range(fulldata_size):
        # Randomly select an MNIST image from the dataset
        img_index = np.random.randint(len(mnist_dataset))
        mnist_image, _ = mnist_dataset[img_index]
        mnist_image = resize(np.squeeze(np.array(mnist_image)), (image_size,image_size))

        # Modify the MNIST image
        modified_image = np.array(np.where(mnist_image[0] > 0, 1, 0), dtype=np.float32)


        # Convert the modified image to a PyTorch tensor
        modified_image_tensor = torch.tensor(modified_image).unsqueeze(0).float()

        # Pass the modified image through the network to get the outputs of all layers
        layer_outputs = net(modified_image_tensor)
        layer_outputs_tensor = np.squeeze(torch.stack(layer_outputs, dim=1).detach().numpy())
    #     print(layer_outputs_tensor.shape)

        # Append the modified image and all layer outputs to the dataset
        input_images.append(modified_image)
        target_outputs.append(layer_outputs_tensor)

    # Convert the lists to NumPy arrays
    print('noise', noise_kind)
    input_images_clean = np.array(input_images)
    target_outputs_clean = np.array(target_outputs)
    
    if noise_kind == 'None':
        input_images = input_images_clean
        target_outputs = target_outputs_clean
        
    elif noise_kind == 'blur':
        input_images_blur = np.zeros_like(input_images_clean)
        target_outputs_blur = np.zeros_like(target_outputs_clean)
        for img_idx in range(len(input_images_clean)):
            input_images_blur[img_idx,:,:] = apply_blur(input_images_clean[img_idx,:,:])
            for frame in range(target_outputs_clean.shape[1]):
                target_outputs_blur[img_idx,frame,:,:] = apply_blur(target_outputs_clean[img_idx,frame,:,:])
        input_images = input_images_blur
        target_outputs = target_outputs_blur
        
    elif noise_kind == 'SP':
        input_images_SP = np.zeros_like(input_images_clean)
        target_outputs_SP = np.zeros_like(target_outputs_clean)
        for img_idx in range(len(input_images_clean)):
            input_images_SP[img_idx,:,:] = apply_salt_and_pepper(input_images_clean[img_idx,:,:])
            for frame in range(target_outputs_clean.shape[1]):
                target_outputs_SP[img_idx,frame,:,:] = apply_salt_and_pepper(target_outputs_clean[img_idx,frame,:,:])
        input_images = input_images_SP
        target_outputs = target_outputs_SP
    print(input_images.shape)
    
    # Plot input data and targets
    try:
        random_integers = random.sample(range(dataset_size), 3)
    except:
        random_integers = random.sample(range(dataset_size), 1)
    num_rows = len(random_integers)
    num_cols = time + 1  # Input image + frames

    plt.figure(figsize=(10, 6))
    for idx, i in enumerate(random_integers):
        print(i)
        plt.subplot(num_rows, num_cols, idx * num_cols + 1)
        plt.imshow(input_images[i, :, :])
        plt.title(f'input image {i}')

        for j in range(1, time + 1):
            plt.subplot(num_rows, num_cols, idx * num_cols + j + 1)
            plt.imshow(target_outputs[i, j - 1, :, :])
            plt.title(f'frame {j}')

    # Save the figure into the "data" folder
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'Data_Noise_{noise_kind}.png'))

    
    #save data
    train_input_name = 'TrainInput_' + str(folder_path) 
    train_target_name = 'TrainTarget_' + str(folder_path)
    val_input_name = 'ValInput_' + str(folder_path)
    val_target_name = 'ValTarget_' + str(folder_path)
    
    # Save the NumPy arrays into the "data" folder
    train_input_path = os.path.join(folder_path, f'{train_input_name}.npy')
    train_target_path = os.path.join(folder_path, f'{train_target_name}.npy')
    val_input_path = os.path.join(folder_path, f'{val_input_name}.npy')
    val_target_path = os.path.join(folder_path, f'{val_target_name}.npy')

    np.save(train_input_path, input_images[:train_size])
    np.save(train_target_path, target_outputs[:train_size])
    np.save(val_input_path, input_images[train_size:])
    np.save(val_target_path, target_outputs_clean[train_size:])
    
    return input_images, target_outputs




if __name__ == '__main__':
    main()
