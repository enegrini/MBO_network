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

    Z = np.zeros_like(X, dtype=int)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = generate_circle_indicator(X[i, j], Y[i, j], radius, center_x, center_y)
    return Z.astype(float)


#define activation
#for datageneration heaviside is fine, but we cannot use it in training.
def smooth_step_function(xx, threshold, steepness=100):
    denominator = 1 + torch.exp(-steepness * (xx - threshold))
    return 1 / denominator

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

