import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import ipywidgets as widgets

# Load the .npz file
data = np.load('public_data.npz', allow_pickle=True)

images = data['data']
labels = data['labels']

i = 0
for image in images: 
    # Normalize image pixel values to a float range [0, 1]
    images[i] = (images[i] / 255).astype(np.float32)
    # Convert image from BGR to RGB
    images[i] = images[i][...,::-1]
    i = i+1
    if (i % 500 == 0):
        print("Processing image: ", i, "\n")
print("Finished processing images")

# Extract image keys
image_keys = list(data.files)

# Define a function to display images
""" def display_images(batch_id):
    plt.figure(figsize=(8, 6))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[batch_id * 5 + i])
        plt.title(labels[batch_id * 5 + i])
        plt.axis('off')
    plt.show()
 """

# Display all images in images array. Show 50 images in the window at a time, and make navigation buttons to scroll through the images
num_images = len(images)    # Number of images
num_cols = int(np.ceil(np.sqrt(num_images)))    # Number of columns in the grid
num_rows = int(np.ceil(num_images / num_cols))    # Number of rows in the grid
num_img = 50    # Number of images to display in the window at a time
# Show the images

show_images = False
if show_images:
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

for label in labels:
    print(label)
    print('\n')

# Create an interactive slider
""" interact(display_images, batch_id=(0, len(image_keys)//5, 1)) """

