"""Code for generating data for Threshold Dynamics on MNIST"""
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
import torch.nn.functional as F
import imageio
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
import os

# Set the device (CPU or GPU) for data generation cpu is fine
device = "cpu"
print('device = ', device)

def main():
    """Function to get results on test data for MBO network"""
    parser = argparse.ArgumentParser(description='Validation for learned kernels and thresholds')
    parser.add_argument('--folder_path', type=str, default='K31standard_T7dT1.5_Noise_None', help='Path to the folder containing data and learned kernels')
    parser.add_argument('--kernel_size', type=int, default=31, help='Size of the kernel')
    parser.add_argument('--GT_path', type=str, default='K31MNIST4id56422_T7dT0.1_thr0.5_Noise_None', help='path to ground truth non-noisy targets')
    
    args = parser.parse_args()
    validation(args.folder_path, args.kernel_size, args.GT_path)


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
    
#define activation
#use heaviside even though training was done with sigmoid

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
    
def validation(folder_path='K31hard_T7dT1.5_Noise_None', kernel_size =31, GT_path = 'K31MNIST4id56422_T7dT0.1_thr0.5_Noise_None'):
    leraned_kernels = folder_path +'/learned_KT/kernels_'+str(kernel_size) +'.npy'
    learned_thresholds= folder_path +'/learned_KT/thresholds_'+str(kernel_size) +'.npy'
    true_kernel_path = 'K31standardM13S6_T7dT1.5_thr0.5_Noise_SP/kernel_true.npy'#folder_path +'/kernel_true.npy'
    input_file = folder_path +'/ValInput.npy'#folder_path +'/ValInput_'+folder_path+'.npy'
    target_file = GT_path +'/ValTarget.npy'#GT_path +'/ValTarget_'+GT_path+'.npy' #folder_path +'/ValTarget_'+folder_path+'.npy'
    input_images = np.load(input_file)
    target_outputs = np.load(target_file)
    leraned_kernels = np.load(leraned_kernels)
    true_kernel = np.load(true_kernel_path)
    learned_thresholds = np.load(learned_thresholds)
    learned_kernel =leraned_kernels[-1] #pick kernel generated with largest steepness parameter
    learned_threshold = learned_thresholds[-1] #pick threshold generated with largest steepness parameter
   
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    # Plot the learned kernel
    im1 = axes[0].imshow(learned_kernel)
    axes[0].set_title('Learned Kernel')
    axes[0].axis('off')  # Remove axis
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # Colorbar for learned kernel
    # Plot the true kernel
    im2 = axes[1].imshow(true_kernel)
    axes[1].set_title('True Kernel')
    axes[1].axis('off')  # Remove axis
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)  # Colorbar for true kernel
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'learned_vs_true_kernel' + str(kernel_size) + '.png'))

    # Print the learned threshold
    print("Learned Threshold:",learned_threshold)
    file_nameT = "threshold."+str(kernel_size) +"txt"
    full_file_path = os.path.join(folder_path, file_nameT)
    with open(full_file_path, "w") as file:
        file.write(f"Learned Threshold = {learned_threshold}\n")

    
    # Convert the modified image to a PyTorch tensor
    image_tensor = torch.tensor(input_images).unsqueeze(1).float().cpu()

    time = target_outputs.shape[1]
    print(time)# This is the number of frames
    net = FlexibleConvNet(num_layers=time, kernel=learned_kernel, threshold=learned_threshold, activation = heaviside).to(device)

    # Pass the modified image through the network to get the outputs of all layers
    layer_outputs = net(image_tensor)
    layer_outputs_numpy = np.squeeze(torch.stack(layer_outputs, dim=1).detach().numpy())
    
    #plot some true and generated videos
    random_integers =[0,1,2,3]#random.sample(range(10), 3)
    for k in random_integers:
        # Create a single figure with three rows
        fig, axs = plt.subplots(3, time, figsize=(1.8*time, 5))

        # Plot the true frames
        for j in range(time):
            axs[0, j].imshow(target_outputs[k, j, :, :])
            axs[0, j].set_title(f'true frame {j + 1}')
            axs[0, j].axis('off')

        # Plot the predicted frames
        for j in range(time):
            axs[1, j].imshow(layer_outputs_numpy[k, j, :, :])
            axs[1, j].set_title(f'predicted frame {j + 1}')
            axs[1, j].axis('off')

        # Plot the difference frames
        for j in range(time):
            axs[2, j].imshow(abs(target_outputs[k, j, :, :] - layer_outputs_numpy[k, j, :, :]))
            axs[2, j].set_title(f'difference frame {j + 1}')
            axs[2, j].axis('off')

        # Adjust spacing and save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, f'combined_plot_{kernel_size}_{k}.png'))  # You can adjust the filename as needed
        plt.close()  # Close the figure after saving to free up resources

    ########################################################################################
    MSE_videos = []  # Store loss values computed using relative MSE
    SSIM_values = []  # Store SSIM values
    Jaccard_indexes = []  # Store Jaccard Index values

    for vid in range(target_outputs.shape[0]):
        target_frame = target_outputs[vid, :, :, :]
        predicted_frame = layer_outputs_numpy[vid, :, :, :]

        # Add a small epsilon to handle division by zero in both MSE and SSIM
        target_frame += 1e-8
        predicted_frame += 1e-8

        # Calculate relative MSE
        MSE_per_video = np.mean(np.abs(target_frame - predicted_frame)**2) / np.mean(abs(target_frame)**2)
        MSE_videos.append(MSE_per_video)

        # Calculate SSIM with data_range parameter
        ssim_value, _ = ssim(target_frame, predicted_frame, full=True, data_range=1)
        SSIM_values.append(ssim_value)

        # Threshold binary images if needed (assuming pixel values are in the range [0, 1])
        threshold = 0.5  # Adjust this threshold based on your specific binary images
        target_binary = (target_frame > threshold).astype(int)
        predicted_binary = (predicted_frame > threshold).astype(int)

        # Calculate Jaccard Index
        intersection = np.logical_and(target_binary, predicted_binary).sum()
        union = np.logical_or(target_binary, predicted_binary).sum()

        # Handle the case where union is zero to avoid division by zero
        jaccard_index = intersection / union if union != 0 else 0
        Jaccard_indexes.append(jaccard_index)

    MSE_total = np.mean(MSE_videos)
    SSIM_total = np.mean(SSIM_values)
    Jaccard_total = np.mean(Jaccard_indexes)

    # Format as exponential with three digits
    formatted_rmse_total = f"{MSE_total * 100:.3f}%"
    formatted_ssim_total = f"{SSIM_total:.3f}"
    formatted_jaccard_total = f"{Jaccard_total * 100:.3f}%"

    print("Relative MSE Kernel = ", formatted_rmse_total)
    print("SSIM Value = ", formatted_ssim_total)
    print("Jaccard Index = ", formatted_jaccard_total)

    # Define the file name where you want to save the metrics
    file_name = "metrics_output" + str(kernel_size) + ".txt"
    full_file_path = os.path.join(folder_path, file_name)

    # Open the file in write mode and save the metrics
    with open(full_file_path, "w") as file:
        file.write(f"Relative MSE Kernel = {formatted_rmse_total}\n")
        file.write(f"SSIM Value = {formatted_ssim_total}\n")
        file.write(f"Jaccard Index = {formatted_jaccard_total}\n")

    return MSE_total, SSIM_total, Jaccard_total
    
if __name__ == '__main__':
    main()
    
