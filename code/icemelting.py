# -*- coding: utf-8 -*-
"""ICEMELTING.ipynb

Automatically generated by Colab.

"""
# Import libraries for image processing
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# # Mounted at Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Paths: import and download from Google Drive - replace with your own directory
image_folder_path = ''
output_folder_path = ''

# Ensure output folder exists
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# List all files and filter only .jpg files (case-insensitive check)
image_filenames = sorted([f for f in os.listdir(image_folder_path) if f.lower().endswith('.jpg')])

# Parameters to crop images with a Cartesian Coordinate (adjustable starting point)
crop_x = 250  # Starting x-coordinate
crop_y = 1000  # Starting y-coordinate
crop_size = 700  # Size of the cropped square (700x700 pixels in this case)

# Function to detect red edges, process each image, and apply the requested transformations
def process_image(image_path):
    print(f"Processing image: {image_path}")

    # Load the image
    image = cv2.imread(image_path)

    # Check if the image is loaded correctly
    if image is None:
        print(f"Error: Could not load image {image_path}. Skipping this file.")
        return

    # Crop the image
    cropped_image = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    # Save the cropped image
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    cropped_image_name = f"{name}_cropped{ext}"
    cropped_image_path = os.path.join(output_folder_path, cropped_image_name)
    cv2.imwrite(cropped_image_path, cropped_image)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Define the range for detecting red color in HSV space
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for the red color in both ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the masks to get the full range of red
    red_mask = mask1 + mask2

    # Use the mask to isolate red regions in the cropped image
    red_regions = cv2.bitwise_and(cropped_image, cropped_image, mask=red_mask)

    # Convert the red regions to grayscale for edge detection
    gray_red_regions = cv2.cvtColor(red_regions, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using Canny
    red_edges = cv2.Canny(gray_red_regions, 50, 150)

    # Find contours from the detected red edges
    contours, _ = cv2.findContours(red_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with a black background
    mask = np.zeros_like(gray_red_regions)

    # Draw the detected contours with white on the mask (edges only)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # The mask already has black inside and white on the edges, so no inversion is needed
    final_output = mask

    # Save the processed image
    new_name = f"{name}_processed{ext}"
    output_image_path = os.path.join(output_folder_path, new_name)
    cv2.imwrite(output_image_path, final_output)

    # Display the processed image steps (for debugging/visualization)
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')

    plt.subplot(1, 4, 2)
    plt.imshow(red_mask, cmap='gray')
    plt.title('Red Color Mask')

    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Filled Inside Black, Outside White')

    plt.subplot(1, 4, 4)
    plt.imshow(final_output, cmap='gray')
    plt.title('Final Inverted Image (White Shape in Center)')

    plt.show()

# Process all images in the folder in name order
for image_filename in image_filenames:
    image_path = os.path.join(image_folder_path, image_filename)
    process_image(image_path)

# Import libraries for video saving
import cv2
import os
import numpy as np

# Paths: Google Drive folder that saved the images - replace with your own directory
output_folder_path = ''

# Compile the processed images into a video
processed_images = [f for f in os.listdir(output_folder_path) if 'processed' in f.lower() and f.lower().endswith('.jpg')]
processed_images.sort()  # Sort by filename to keep the order of ice-melting

video_frames = []  # List to store video frames for saving as .npy

if processed_images:
    video_output_path = os.path.join(output_folder_path, 'icemelting_video.mp4')
    npy_output_path = os.path.join(output_folder_path, 'icemelting_video.npy')
    frame_rate = 10  # Adjust frames as needed

    first_image_path = os.path.join(output_folder_path, processed_images[0])
    first_image = cv2.imread(first_image_path)
    height, width = first_image.shape[:2]

    video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for image_filename in processed_images:
        img_path = os.path.join(output_folder_path, image_filename)
        img = cv2.imread(img_path)
        video_writer.write(img)
        video_frames.append(img)  # Append the image frame to the list

    video_writer.release()
    print("Video creation complete. Video is saved as 'icemelting_video.mp4'.")

    # Save the video frames as a .npy file
    np.save(npy_output_path, np.array(video_frames))
    print("Video frames saved as 'icemelting_video.npy'.")
else:
    print("No processed images found. Video creation skipped.")
