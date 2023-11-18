import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('public_data.npz', allow_pickle=True)

images = data['data']
labels = data['labels']

print("Number of images: ", len(images))

i = 0
for image in images: 
    # Normalize image pixel values to a float range [0, 1]
    images[i] = (images[i] / 255).astype(np.float32)
    # Convert image from BGR to RGB
    #images[i] = images[i][...,::-1]
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

# Sanitize data
positions_to_remove = [58, 95, 137, 138, 171, 207, 338, 412, 434, 486, 506, 529, 571, 
                           599, 622, 658, 692, 701, 723, 725, 753, 779, 783, 827, 840, 880, 
                           898, 901, 961, 971, 974, 989, 1028, 1044, 1064, 1065, 1101, 1149, 
                           1172, 1190, 1191, 1265, 1268, 1280, 1333, 1384, 1443, 1466, 1483, 
                           1528, 1541, 1554, 1594, 1609, 1630, 1651, 1690, 1697, 1752, 1757,
                           1759, 1806, 1828, 1866, 1903, 1938, 1939, 1977, 1981, 1988, 2022, 
                           2081, 2090, 2150, 2191, 2192, 2198, 2261, 2311, 2328, 2348, 2380, 
                           2426, 2435, 2451, 2453, 2487, 2496, 2515, 2564, 2581, 2593, 2596, 
                           2663, 2665, 2675, 2676, 2727, 2734, 2736, 2755, 2779, 2796, 2800, 
                           2830, 2831, 2839, 2864, 2866, 2889, 2913, 2929, 2937, 3033, 3049, 
                           3055, 2086, 3105, 3108, 3144, 3155, 3286, 3376, 3410, 3436, 3451,
                           3488, 3490, 3572, 3583, 3666, 3688, 3700, 3740, 3770, 3800, 3801, 
                           3802, 3806, 3811, 3821, 3835, 3862, 3885, 3896, 3899, 3904, 3927, 
                           3931, 3946, 3950, 3964, 3988, 3989, 4049, 4055, 4097, 4100, 4118, 
                           4144, 4150, 4282, 4310, 4314, 4316, 4368, 4411, 4475, 4476, 4503,
                           4507, 4557, 4605, 4618, 4694, 4719, 4735, 4740, 4766, 4779, 4837,
                           4848, 4857, 4860, 4883, 4897, 4903, 4907, 4927, 5048, 5080, 5082, 
                           5121, 5143, 5165, 5171]
n = 0
for pos in positions_to_remove:
    new_pos = pos - n
    print("Removing image at position: ", pos, " - New Position is ", new_pos)
    images = np.delete(images, new_pos, axis=0)
    labels = np.delete(labels, new_pos, axis=0)
    n = n + 1

# Display all images in images array. Show 50 images in the window at a time, and make navigation buttons to scroll through the images
num_images = len(images)    # Number of images
num_cols = int(np.ceil(np.sqrt(num_images)))    # Number of columns in the grid
num_rows = int(np.ceil(num_images / num_cols))    # Number of rows in the grid
#num_img = 50    # Number of images to display in the window at a time
# Show the images

show_all_images = False
num_images = len(images)    # Number of images
if show_all_images:
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.ion()
    plt.show()
else:
    steps = 100
    counter = 0
    num_total_images = len(images)
    num_cols = int(np.ceil(np.sqrt(steps)))    # Number of columns in the grid
    num_rows = int(np.ceil(steps / num_cols))    # Number of rows in the grid
    while (counter < num_total_images):
        for i in range (counter, counter + steps):
            plt.subplot(num_rows, num_cols, i+1 - counter)
            plt.imshow(images[i])
            plt.axis('off')
            plt.text(0, 0, f"{i} - {labels[i]}", fontsize=8, color='red')
        counter = counter + steps
        plt.show()

for label in labels:
    print(label)
    print('\n')

# Create an interactive slider
""" interact(display_images, batch_id=(0, len(image_keys)//5, 1)) """

