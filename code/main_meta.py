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
torch.autograd.set_detect_anomaly(True)
from metalearning import *

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def main():
    """Function to run training and testing of metalearning MBO network"""
    # Get the current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = 'results/Cropped_fire_meta' + timestamp
    os.makedirs(results_folder, exist_ok=True)
    
    # Set parameters
    data_folder = "Selected_ice_meta" #'video_data/data_Final_blur'
    GT_folder = "Selected_ice_meta" #'video_data/data_Final_noNoise'
    n_examples =27 # Number of examples per dataset for training
    model_save_path = results_folder + f'/trained_model_{timestamp}'
    loss_plot_path = results_folder + f'/training_loss_plot_{timestamp}.png'
    test_loss_plot_path = results_folder + f'/testing_loss_plot_{timestamp}.png'
    kernel_plot_save_path = results_folder + f'/learned_vs_true_kernels_{timestamp}.png'
    video_plot_save_path =  results_folder + f'/learned_vs_true_videos_{timestamp}.png'
    kernel_folder = 'video_data/true_kernels_Final_blur'
    threshold_folder = 'video_data/true_thresholds_Final_blur'
    num_epochs = 500
    learning_rate = 1e-5
    batch_size = 1
    kernel_size = 51
    time_input = 4
    device = 'cuda:1'
    print_ep = 100
    model_path="results/K51_blur2024-10-09_14-29-49/trained_model_2024-10-09_14-29-49.pth"
    training = False
    fires = True

    if training:
        use_saved_model = False
        model_path = None
        # # Load training and testing videos
        videos = get_train_videos(data_folder, n_examples)
        print('training videos shape:', videos.shape)
        videos_test_plot, videos_valid = get_test_videos(data_folder, n_examples)
        GT_videos_plot, GT_videos_valid = get_test_videos(GT_folder, n_examples)
        print('testing videos shape:', videos_valid.shape)
    
        # # Training
        print("Starting training...")
        loss_epoch, trained_model = training_loop(
            videos, NewCNN, FlexibleConvNet, multilayer_loss, print_ep=print_ep, device=device,
            num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size,
            kernel_size=kernel_size, time_input=time_input, model_save_path=model_save_path,
            loss_plot_path=loss_plot_path
        )
    
        print("Starting testing...")  
        mse_total, jaccard_total, ssim_total = testing_function(
            NewCNN=trained_model, FlexibleConvNet=FlexibleConvNet, 
            videos_test_plot=videos_test_plot, videos_valid=videos_valid, GT_videos_plot=GT_videos_plot, GT_videos_valid=GT_videos_valid,
            kernel_folder=kernel_folder, threshold_folder=threshold_folder, device=device,
            time_input=time_input, loss_plot_save_path=test_loss_plot_path, kernel_plot_save_path=kernel_plot_save_path,
            video_plot_save_path = video_plot_save_path, use_saved_model=use_saved_model, model_path=model_path
        )
    
        print(f"Testing completed.\nMean Relative MSE: {mse_total}\nMean Jaccard Index: {jaccard_total}, \nMean SSIM: {ssim_total}")

    else:
        use_saved_model = True
        if not fires:
            videos_test_plot, videos_valid = get_test_videos(data_folder, n_examples)
            GT_videos_plot, GT_videos_valid = get_test_videos(GT_folder, n_examples)
            print(videos_valid.shape)
        else:
            videos_test_plot =  np.load('Selected_ice_meta/ValTarget.npy')
            print(videos_test_plot.shape)
            videos_valid = np.load('Selected_ice_meta/ValTarget.npy')
            GT_videos_plot = videos_test_plot
            GT_videos_valid = videos_valid
        # Testing
        if use_saved_model:
            trained_model=NewCNN(input_channels=1, output_size=kernel_size, num_frames_input=time_input, 
                         height=videos_test_plot.shape[2], width=videos_test_plot.shape[3], device=device).to(device)
        print("Starting testing...")  
        mse_total, jaccard_total, ssim_total = testing_function(
            NewCNN=trained_model, FlexibleConvNet=FlexibleConvNet, 
            videos_test_plot=videos_test_plot, videos_valid=videos_valid, GT_videos_plot=GT_videos_plot, GT_videos_valid=GT_videos_valid,
            kernel_folder=kernel_folder, threshold_folder=threshold_folder, device=device,
            time_input=time_input, loss_plot_save_path=test_loss_plot_path, kernel_plot_save_path=kernel_plot_save_path,
            video_plot_save_path = video_plot_save_path, use_saved_model=use_saved_model, model_path=model_path
        )
    
        print(f"Testing completed.\nMean Relative MSE: {mse_total}\nMean Jaccard Index: {jaccard_total}\nMean SSIM: {ssim_total}")
        

if __name__ == "__main__":
    main()





