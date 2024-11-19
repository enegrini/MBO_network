"""Code for Threshold Dynamics on MNIST"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from datetime import datetime, timedelta

import time
# import imageio

import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
# from skimage.metrics import structural_similarity as ssim

import torch.optim as optim
import numpy as np
from PIL import Image
# Set the device (CPU or GPU)

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from torch.utils.data import random_split
from skimage.metrics import structural_similarity as ssim

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('device = ', device)

class FlexibleConvNet(nn.Module):
    """Implements MBO algorithm"""
    def __init__(self, num_layers, kernel, threshold, activation, steepness=1e2, device='cuda:1'):
        super(FlexibleConvNet, self).__init__()
        self.num_layers = num_layers
        self.shared_kernel = kernel.unsqueeze(0)
        self.shared_threshold = threshold
        self.activation = activation
        self.steepness = steepness
        
    def forward(self, x):
        layer_outputs = []  # List to store the output of each layer
        for i in range(self.num_layers):
            # Apply 2D convolution with 'same' padding using the shared kernel
            x = nn.functional.conv2d(x, self.shared_kernel, padding='same')
            # Apply a custom activation function with the shared threshold
            x = self.activation(x, self.shared_threshold, self.steepness)
            
            layer_outputs.append(x)  # Append the output of the current layer to the list
        return torch.cat(layer_outputs, dim =1)  # Return the list of layer outputs


class NewCNN(nn.Module):
    """CNN to learn kernel and threshold"""
    def __init__(self, input_channels, output_size, num_frames_input, height, width, device='cuda:1'):
        super(NewCNN, self).__init__()
        self.output_size = output_size
        
        self.conv1 = nn.Conv2d(in_channels=input_channels * num_frames_input, out_channels=64, kernel_size=15, stride=1, padding=7)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=15, stride=1, padding=7)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=15, stride=1, padding=7)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(256 * (height // 8) * (width // 8), 512)
        self.fc2_kernel = nn.Linear(512, output_size * output_size)
        self.fc2_threshold = nn.Linear(512, 1)
        
        self.activ = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool(self.activ(self.conv1(x)))
        x = self.pool(self.activ(self.conv2(x)))
        x = self.pool(self.activ(self.conv3(x)))
        # Apply more convolutional layers as needed

        # Flatten the tensor
        x = x.flatten(start_dim=1)

        # Apply the first feedforward layer
        x = self.activ(self.fc1(x))

        # Predict the kernel
        kernel = self.fc2_kernel(x).view(-1, self.output_size, self.output_size).to(x.device)
#         normalize kernel
        reshaped_kernel = kernel.view(kernel.size(0), -1)
        softmax_kernel = torch.nn.functional.softmax(reshaped_kernel, dim=-1)
        softmax_kernel = softmax_kernel.view(kernel.size())
        
        threshold = self.fc2_threshold(x)
        threshold = self.sigmoid(threshold)
        
        return softmax_kernel,threshold 


def heaviside(x, threshold=0.0,steepness = None):
    """
    Heaviside step function with a specified threshold.

    Args:
        x (torch.Tensor): Input tensor.
        threshold (float): Threshold value.

    Returns:
        torch.Tensor: Heaviside step function with the specified threshold.
    """
    return torch.where(x.to(device) > threshold, torch.tensor(1.0, dtype=x.dtype).to(device), torch.tensor(0.0, dtype=x.dtype).to(device))

def smooth_step_function(xx, threshold, steepness=100):
    denominator = 1 + torch.exp(-steepness * (xx - threshold))
    return 1 / denominator

def get_train_videos(data_folder, n_examples):
    """
    INPUT:
    - data_folder: path to the folder containing the .npy files
    - n_examples: number of videos per dataset
    
    OUTPUT:
    - concatenated_videos: videos for training (all frames for now, but only a few selected at training time)
    """
    # Get all .npy files in the directory
    all_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy')])
    
    # Select the files based on the given indices
    # selected_files = [all_files[i] for i in dataset_indices]
    
    videos_list = []
    for file_name in all_files:
        dataset_path = os.path.join(data_folder, file_name)
        videos = np.load(dataset_path,allow_pickle=False)
        videos_list.append(videos[:n_examples])
    
    concatenated_videos = np.concatenate(videos_list, axis=0)
    print('train data shape', concatenated_videos.shape)
    
    return concatenated_videos


def get_test_videos(data_folder, n_examples):
    """
    INPUT:
    - data_folder: path to the folder containing the .npy files
    - n_examples: number of videos per dataset
    
    OUTPUT:
    - videos_test_plot: one video per class for plots
    - videos_valid: videos for validation errors
    """
    # Get all .npy files in the directory
    all_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy')])
    
    # Select the files based on the given indices
    # selected_files = [all_files[i] for i in dataset_indices]
    
    # Select videos for test plotting and validation
    videos_test_plot = []
    videos_valid = []
    for file_name in all_files:
        dataset_path = os.path.join(data_folder, file_name)
        videos = np.load(dataset_path)
        
        # Randomly select one video for test plotting
        j = 1#random.randint(n_examples, len(videos) - 1)
        videos_test_plot.append(videos[j:j+1])
        
        # Remaining videos for validation
        videos_valid.append(videos[n_examples:]) #30 just because we only have total of 30 fires
    
    videos_test_plot = np.concatenate(videos_test_plot, axis=0)
    videos_valid = np.concatenate(videos_valid, axis=0)
    
    return videos_test_plot, videos_valid

def multilayer_loss(outputs,targets):
    tot_loss = 0.
    for i in range(targets.shape[1]):
        mse_loss = nn.functional.mse_loss(outputs[:,i,:,:], targets[:,i,:,:]) 
        tot_loss += mse_loss
    return(tot_loss)


def training_loop(videos, NewCNN, FlexibleConvNet, multilayer_loss, print_ep = 50,  device = 'cuda:1',
                  num_epochs=200, learning_rate=1e-4, batch_size=20, kernel_size=31, 
                  time_input=4, model_save_path='trained_model.pth', loss_plot_path='training_loss_plot.png'):
    """
    Training loop for the NewCNN model.
    
    Args:
        videos (np.ndarray): Input videos for training.
        device (torch.device): Device to run the model on ('cuda' or 'cuda:1').
        NewCNN (torch.nn.Module): The neural network model class.
        FlexibleConvNet (torch.nn.Module): Model for the MBO algorithm.
        multilayer_loss (callable): Loss function.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        kernel_size (int): Size of the output from the NewCNN.
        time_input (int): Number of frames used as input.
        model_save_path (str): Path to save the trained model.
        loss_plot_path (str): Path to save the training loss plot.
        
    Returns:
        tuple: A tuple containing the list of losses per epoch and the trained model.
    """
    
    input_channels = 1  # Grayscale input
    
    new_cnn = NewCNN(input_channels=input_channels, output_size=kernel_size, num_frames_input=time_input, 
                     height=videos.shape[2], width=videos.shape[3], device=device).to(device)
    
    criterion = multilayer_loss
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(videos[:, :time_input, :, :], dtype=torch.float),
                                             torch.tensor(videos[:, :time_input, :, :], dtype=torch.float))

    # Define the train-test split ratio, e.g., 80% for training and 20% for testing
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Split the dataset into training and testing datasets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    optimizer = optim.Adam(new_cnn.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5)
    
    new_cnn.train()
    loss_epoch = []
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            output_videos = []
            for i in range(inputs.shape[0]):
                learned_kernel, threshold = new_cnn(inputs[i:i+1, :, :, :])
                video_frames = []
                MBOnet_nextFrame = FlexibleConvNet(num_layers=1, kernel=learned_kernel, threshold=threshold,
                                                   activation=smooth_step_function, steepness=100, device=device).to(device)
                for time_step in range(time_input):
                    current_frame = inputs[i:i+1, time_step:time_step+1, :, :].to(device)
                    next_frame = MBOnet_nextFrame(current_frame)
                    video_frames.append(next_frame)
                current_video = torch.cat(video_frames, dim=1)
                output_videos.append(current_video)
            final_output = torch.cat(output_videos, dim=0)
            
            loss = criterion(final_output[:, :-1, :, :], targets[:, 1:, :, :])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step(total_loss / len(data_loader))
        loss_epoch.append(total_loss / len(data_loader))

        # Compute and print test loss every 50 epochs
        if epoch % print_ep == 0:
            new_cnn.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    output_videos = []
                    for i in range(inputs.shape[0]):
                        learned_kernel, threshold = new_cnn(inputs[i:i+1, :, :, :])
                        video_frames = []
                        MBOnet_nextFrame = FlexibleConvNet(num_layers=1, kernel=learned_kernel, threshold=threshold,
                                                           activation=smooth_step_function, steepness=100, device=device).to(device)
                        for time_step in range(time_input):
                            current_frame = inputs[i:i+1, time_step:time_step+1, :, :].to(device)
                            next_frame = MBOnet_nextFrame(current_frame)
                            video_frames.append(next_frame)
                        current_video = torch.cat(video_frames, dim=1)
                        output_videos.append(current_video)
                    final_output = torch.cat(output_videos, dim=0)
                    
                    loss = criterion(final_output[:, :-1, :, :], targets[:, 1:, :, :])
                    total_test_loss += loss.item()
    
            test_loss = total_test_loss / len(test_loader)
            #remaining training time estimate
            epoch_duration = datetime.now() - epoch_start_time
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining_time = timedelta(seconds=epoch_duration.total_seconds() * remaining_epochs)
            estimated_end_time = datetime.now() + estimated_remaining_time
            
            print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {loss_epoch[-1]:.4f} - Test Loss: {test_loss:.4f}")
            print(f"Estimated Remaining Time: {str(estimated_remaining_time)}")
            print(f"Estimated End Time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

            model_ep_path = model_save_path + 'ep' + str(epoch) +'.pth'
            torch.save(new_cnn.state_dict(), model_ep_path)
            print("Model saved to", model_ep_path)
            
            new_cnn.train()  # Switch back to training mode for the next epoch
       
    
    # Save the trained model
    torch.save(new_cnn.state_dict(), model_save_path + '.pth')
    print("Training complete! Model saved to", model_save_path)
    
    # Plot and save the loss curve
    plt.figure()
    plt.plot(loss_epoch, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")
    
    return loss_epoch, new_cnn


def load_thresholds(file_paths):
    """Load thresholds from .txt files."""
    thresholds = []
    for path in file_paths:
        with open(path, 'r') as f:
            threshold = float(f.readline().strip())
            thresholds.append(threshold)
    return thresholds


def testing_function(NewCNN, FlexibleConvNet, videos_test_plot, videos_valid, GT_videos_plot, GT_videos_valid,
                     kernel_folder, threshold_folder, device='cuda:1', time_input=4, 
                     loss_plot_save_path='testing_loss_plot.png', 
                     kernel_plot_save_path='learned_vs_true_kernels.png',
                     video_plot_save_path ='learned_vs_true_videos.png', 
                     use_saved_model=False, model_path=None):
    """
    Testing function to evaluate the trained model.
    
    Args:
        NewCNN (torch.nn.Module): The neural network model class.
        FlexibleConvNet (torch.nn.Module): Model for the MBO algorithm.
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
        videos_test_plot (np.ndarray): Input videos for testing.
        videos_valid (np.ndarray): Validation videos.
        kernel_folder (str): Path to the folder containing the true kernel images.
        threshold_folder (str): Path to the folder containing the true threshold files.
        time_input (int): Number of frames used as input.
        loss_plot_save_path (str): Path to save the loss plot.
        kernel_plot_save_path (str): Path to save the kernel comparison plot.
        video_plot_save_path (str): Path to save the video comparison plot.
        use_saved_model (bool): Flag to indicate whether to load a saved model.
        model_path (str): Path to the path to the saved model.
        
    Returns:
        tuple: A tuple containing the mean relative MSE and mean Jaccard Index.
    """

    # Load all kernel and threshold paths
    all_kernel_paths = sorted([os.path.join(kernel_folder, f) for f in os.listdir(kernel_folder) if f.endswith('.npy')])
    all_threshold_paths = sorted([os.path.join(threshold_folder, f) for f in os.listdir(threshold_folder) if f.endswith('.txt')])
    true_thresholds = load_thresholds(all_threshold_paths)
    
    videos_tensor = torch.tensor(videos_test_plot).to(device)
    total_frames = videos_tensor.size(1)

    list_testvids = [0]#np.arange(len(videos_tensor))
    try:
        np.random.seed(42)
        test_video_idx =[0,1]#[3,20,41,62,78]#np.random.choice(list_testvids, size=10, replace=False)
        print('test video indices', test_video_idx)
    except:
        np.random.seed(42)
        test_video_idx = [0,1]#[3,20,41,62,78]#np.random.choice(list_testvids, size=10, replace=True)
        print('test video indices', test_video_idx)

    if use_saved_model:
        assert model_path is not None, "model_path must be specified if use_saved_model is True."
        NewCNN.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        NewCNN.to(device)
        NewCNN.eval()
        print(f"Loaded model from {model_path}")
    
    # Plot the learned and true kernels
    fig, axs = plt.subplots(2, len(test_video_idx), figsize=(15, 6))
    for i, idx in enumerate(test_video_idx):
        true_kernel_path = all_kernel_paths[idx]
        true_kernel = np.load(true_kernel_path)
        img1=axs[0, i].imshow(true_kernel)
        fig.colorbar(img1, ax=axs[0, i], fraction=0.046, pad=0.04)
        axs[0, i].axis('off')
    
        # Use idx to select the corresponding test video
        test_video = videos_tensor[idx:idx+1, :, :, :]
        learned_kernel, learned_threshold = NewCNN(test_video[:, :time_input, :, :].to(device))
        print(f'True threshold {idx}: {true_thresholds[idx]}')
        print(f'Learned threshold {idx}: {learned_threshold.item()}')
    
        img2 = axs[1, i].imshow(learned_kernel.squeeze(0).detach().cpu().numpy())
        axs[1, i].set_title(f'Learned Kernel {i+1}')
        fig.colorbar(img2, ax=axs[1, i], fraction=0.046, pad=0.04)
        axs[1, i].axis('off')

    # for ax_row in axs[1:]:
    #     for ax in ax_row:
    #         ax.axis('off')
# 
    plt.tight_layout()
    plt.savefig(kernel_plot_save_path)
    plt.close()
    print(f"Kernels comparison plot saved to {kernel_plot_save_path}")

    # Create a single figure to plot all generated and test frames
    num_videos = len(test_video_idx)
    num_frames = videos_tensor.size(1) - 1#1  # Number of frames per video
    print(num_frames)
    
    # Create a figure with 2*num_videos rows and num_frames columns
    fig, axs = plt.subplots(2 * num_videos, num_frames, figsize=(num_frames * 1.8, 2 * num_videos * 1.5))
    
    for i, idx in enumerate(test_video_idx):
        test_video = videos_tensor[idx:idx+1, :, :, :]
        GT_video = GT_videos_plot[idx:idx+1, 1:, :, :]
        learned_kernel, learned_threshold = NewCNN(test_video[:, :time_input, :, :].to(device))
        video_generatorMBO = FlexibleConvNet(num_layers=6, kernel=learned_kernel, threshold=learned_threshold, activation=heaviside, device=device).to(device)
        generated_video = video_generatorMBO(test_video[:, 0:1, :, :])[:,:,:,:]
        
        # Plotting generated video frames
        for j in range(num_frames):
            axs[2 * i, j].imshow(generated_video[0, j].detach().cpu().numpy())
            axs[2 * i, j].set_title(f'Generated frame {j+2}')
            axs[2 * i, j].axis('off')
        
        # Plotting test video frames
        for j in range(num_frames):
            axs[2 * i + 1, j].imshow(GT_video[0, j])
            axs[2 * i + 1, j].set_title(f'Test frame {j+2}')
            axs[2 * i + 1, j].axis('off')

    plt.tight_layout()
    plt.savefig(video_plot_save_path)
    plt.close()
    print(f"Generated vs Test frames plot saved to {video_plot_save_path}")

    MSE_videos = []
    Jaccard_indexes = []
    SSIM_scores = []

    videos_valid_tensor = torch.tensor(videos_valid).to(device)
    total_frames =4# videos_valid_tensor.size(1) - 1
    print(total_frames)

    for i in range(len(videos_valid_tensor)):
        test_video = videos_valid_tensor[i:i+1, :, :, :]
        GT_video_valid = GT_videos_valid[i:i+1, 1:, :, :]
        learned_kernel, learned_threshold = NewCNN(test_video[:, :time_input, :, :])
        video_generatorMBO = FlexibleConvNet(num_layers=6, kernel=learned_kernel, threshold=learned_threshold, activation=heaviside, device=device).to(device)
        generated_video = video_generatorMBO(test_video[:, 0:1, :, :])[:,:,:,:]

        target_frame = GT_video_valid[:, :, :, :] + 1e-8
        predicted_frame = generated_video.detach().cpu().numpy() + 1e-8

        MSE_per_video = np.mean(np.abs(target_frame - predicted_frame)**2) / np.mean(abs(target_frame)**2)
        if MSE_per_video<=1.1:
            MSE_videos.append(MSE_per_video)

        threshold_img = 0.5
        # Jaccard Index computation per frame
        for j in range(target_frame.shape[1]):
            target_binary = (target_frame[:, j, :, :] > threshold_img).astype(int)
            predicted_binary = (predicted_frame[:, j, :, :] > threshold_img).astype(int)

            intersection = np.logical_and(target_binary, predicted_binary).sum(axis=(1, 2))
            union = np.logical_or(target_binary, predicted_binary).sum(axis=(1, 2))

            jaccard_index_per_frame = intersection / np.where(union != 0, union, 1)
            Jaccard_indexes.append(np.mean(jaccard_index_per_frame))
        # target_binary = (target_frame > threshold).astype(int)
        # predicted_binary = (predicted_frame > threshold).astype(int)

        # intersection = np.logical_and(target_binary, predicted_binary).sum()
        # union = np.logical_or(target_binary, predicted_binary).sum()

        # jaccard_index = intersection / union if union != 0 else 0
        # Jaccard_indexes.append(jaccard_index)

        # SSIM computation
        ssim_per_frame = []
        for j in range(target_frame.shape[1]):
            ssim_value, _ = ssim(target_frame[:, j, :, :].squeeze(), predicted_frame[:, j, :, :].squeeze(),data_range=1.0, full=True)
            ssim_per_frame.append(ssim_value)
        SSIM_scores.append(np.mean(ssim_per_frame))

    MSE_total = np.mean(MSE_videos)
    Jaccard_total = np.mean(Jaccard_indexes)
    SSIM_total = np.mean(SSIM_scores)

    formatted_rmse_total = f"{MSE_total * 100:.3f}%"
    formatted_jaccard_total = f"{Jaccard_total * 100:.3f}%"
    formatted_ssim_total = f"{SSIM_total * 100:.3f}%"

    print("Relative MSE = ", formatted_rmse_total)
    print("Jaccard Index = ", formatted_jaccard_total)
    print("SSIM = ", formatted_ssim_total)

    return MSE_total, Jaccard_total, SSIM_total


