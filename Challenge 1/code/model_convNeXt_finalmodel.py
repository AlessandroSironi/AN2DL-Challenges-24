"""
AN2DL - Homework 1
Image Classification

Group Name: TensorFlex
Sironi Alessandro, Stefanizzi Giovanni, Stefanizzi Tomaso, Villa Ismaele

"""
# Fix randomness and hide warnings
seed = 69420

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
np.random.seed(seed)

import logging

import random
random.seed(seed)

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from tensorflow.keras.applications.mobilenet import preprocess_input
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

print("Finished loading libraries")

model_name = "model_convNeXt_bestsub_18nov"

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, model_name))

    def predict(self, X):
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)  # Shape [BS]

        return out


# The `show_images` function takes a list of images as input and displays a grid of up to 10 images using Matplotlib. 
# The number of columns in the grid is determined by the square root of the total number of images (rounded up),
# and the number of rows is calculated accordingly. The function then iterates through the images, plots each one in a subplot 
# of the grid, turns off axis labels, and finally displays the entire grid of images using Matplotlib's `plt.show()` function.
def show_images(images):
    num_images = 10    # Number of images tp plot
    num_cols = int(np.ceil(np.sqrt(num_images)))    # Number of columns in the grid
    num_rows = int(np.ceil(num_images / num_cols))  # Number of rows in the grid

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

# The `plot_results` function visualizes training history for a neural network, displaying plots for loss and accuracy during
# training and validation. It also highlights the epoch with the highest validation accuracy using a marker in the accuracy plot.
# The function uses Matplotlib for plotting.
def plot_results(history): #TODO: check if it works
    print("Plotting results...")
    # Plot the training
    plt.figure(figsize=(15,5))
    plt.plot(history['loss'], alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_loss'], label='Vanilla CNN', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['accuracy'], alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_accuracy'], label='Vanilla CNN', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)

    plt.show()

    # Find the epoch with the highest validation accuracy
    best_epoch = np.argmax(history['val_accuracy'])

    # Plot training and validation performance metrics
    plt.figure(figsize=(20, 5))

    # Plot training and validation loss
    plt.plot(history['loss'], label='Training', alpha=0.8, color='#ff7f0e', linewidth=3)
    plt.plot(history['val_loss'], label='Validation', alpha=0.8, color='#4D61E2', linewidth=3)
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=0.3)

    plt.figure(figsize=(20, 5))

    # Plot training and validation accuracy, highlighting the best epoch
    plt.plot(history['accuracy'], label='Training', alpha=0.8, color='#ff7f0e', linewidth=3)
    plt.plot(history['val_accuracy'], label='Validation', alpha=0.8, color='#4D61E2', linewidth=3)
    plt.plot(best_epoch, history['val_accuracy'][best_epoch], marker='*', alpha=0.8, markersize=10, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=0.3)

    plt.show()

# The 'mse' function returns the 'Mean Squared Error' between two images given as input
def mse(imageA, imageB):
    # The 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # Return the MSE, the lower the error, the more "similar" the two images are
    return err

# The 'sanitize_input' function removes all the outliers from the input dataset
def sanitize_input(images,labels):
    # Delete trolololol and shrek
    # Positions_to_remove contains all the indexes of outliers in the dataset, those have been separatly calculated using the 'mse' function
    # while here we are just checking that no outlier has been missed
    positions_to_remove_old = [58, 95, 137, 138, 171, 207, 338, 412, 434, 486, 506, 529, 571,
                           599, 622, 658, 692, 701, 723, 725, 753, 779, 783, 827, 840, 880,
                           898, 901, 961, 971, 974, 989, 1028, 1044, 1064, 1065, 1101, 1149,
                           1172, 1190, 1191, 1265, 1268, 1280, 1333, 1384, 1443, 1466, 1483,
                           1528, 1541, 1554, 1594, 1609, 1630, 1651, 1690, 1697, 1752, 1757,
                           1759, 1806, 1828, 1866, 1903, 1938, 1939, 1977, 1981, 1988, 2022,
                           2081, 2090, 2150, 2191, 2192, 2198, 2261, 2311, 2328, 2348, 2380,
                           2426, 2435, 2451, 2453, 2487, 2496, 2515, 2564, 2581, 2593, 2596,
                           2663, 2665, 2675, 2676, 2727, 2734, 2736, 2755, 2779, 2796, 2800,
                           2830, 2831, 2839, 2864, 2866, 2889, 2913, 2929, 2937, 3033, 3049,
                           3055, 3086, 3105, 3108, 3144, 3155, 3286, 3376, 3410, 3436, 3451,
                           3488, 3490, 3572, 3583, 3666, 3688, 3700, 3740, 3770, 3800, 3801,
                           3802, 3806, 3811, 3821, 3835, 3862, 3885, 3896, 3899, 3904, 3927,
                           3931, 3946, 3950, 3964, 3988, 3989, 4049, 4055, 4097, 4100, 4118,
                           4144, 4150, 4282, 4310, 4314, 4316, 4368, 4411, 4475, 4476, 4503,
                           4507, 4557, 4605, 4618, 4694, 4719, 4735, 4740, 4766, 4779, 4837,
                           4848, 4857, 4860, 4883, 4897, 4903, 4907, 4927, 5048, 5080, 5082,
                           5121, 5143, 5165, 5171]
    
    positions_to_remove = []
    pos_first_shrek = 58
    pos_first_trolo = 338

    for pos in range(len(images)):
        if (mse(images[pos_first_shrek],images[pos])==0 or mse(images[pos_first_trolo],images[pos])==0):
            positions_to_remove.append(pos)
    if (positions_to_remove != positions_to_remove_old):
        print("ERROR: Different positions to remove")
        exit()
    print("Removing " + str(len(positions_to_remove)) + ' outliers from the dataset...')
    
    n = 0
    # Let's remove those SHreks and Trolololol
    for pos in positions_to_remove:
        new_pos = pos - n
        #print("Removing image at position: ", pos, " - New Position is ", new_pos)
        images = np.delete(images, new_pos, axis=0)
        labels = np.delete(labels, new_pos, axis=0)
        n = n + 1

    return images, labels


# The `train_model` function performs the following tasks in Python:
# 1. Loads image data from a .npz file, applies preprocessing, and encodes labels.
# 2. Splits the dataset into training, validation, and test sets.
# 3. Builds a ConvNeXtLarge model for transfer learning with image augmentation.
# 4. Trains the model on the training set, saves the best model, and reloads it.
# 5. Fine-tunes the model by setting specific layers as trainable and freezing others.
# 6. Compiles and trains the fine-tuned model.
# 7. Optionally plots training history graphs.
# 8. Evaluates the model on the test set, computes and displays classification metrics, and plots a confusion matrix.

def train_model():
    print("[*] Training model ", model_name, "...")
    # Load images from the .npz file
    data_path = 'public_data.npz'
    data = np.load(data_path, allow_pickle=True)
    images = data['data']
    labels = data['labels']

    # ConvNeXt models expect their inputs to be float tensors of pixels with values in the [0-255] range so let's cast the images
    images = (images).astype(np.float32)

    # sanitize input
    images, labels = sanitize_input(images, labels)

    # Let's encode to the labels as we want them to be a 2D array of type [[0,1],[1,0],[1,0],...]
    labels = np.array(labels)
    labels = LabelEncoder().fit_transform(labels)
    labels = tfk.utils.to_categorical(labels,len(np.unique(labels)))

    #  Split the dataset in train set and test set, use the stratify option to maintain the class distribution in the train and test datasets
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.15, stratify=np.argmax(labels, axis=1), random_state=seed)

    # Further split the test set into test and validation sets, stratifying the labels
    images_test, images_val, labels_test, labels_val = train_test_split(images_test, labels_test, test_size=0.9, stratify=np.argmax(labels_test, axis=1), random_state=seed)

    print("Shapes of the sets:")
    print(f"images_train shape: {images_train.shape}, labels_train shape: {labels_train.shape}")
    print(f"images_val shape: {images_val.shape}, labels_val shape: {labels_val.shape}")
    print(f"images_test shape: {images_test.shape}, labels_test shape: {labels_test.shape}")
    print("\n")

    input_shape = images_train.shape[1:]
    output_shape = labels_train.shape[1:]

    # Let's build the model

    # If include_preprocessing=True no preprocessing is needed
    externalNet = tf.keras.applications.ConvNeXtLarge(
        model_name="convnext_large",
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
    )

    #A utomatically get the name of the network
    network_keras_name = externalNet.name
    print("[*] Network name: ", network_keras_name)

    for i, layer in enumerate(externalNet.layers):
        print(i, layer.name, layer.trainable)

    # Use the supernet as feature extractor, i.e. freeze all its weigths
    externalNet.trainable = False

    # Create an input layer with standard shape (96, 96, 3)
    inputs = tfk.Input(shape=(96, 96, 3))

    # Perform image augmentation
    augmentation = tf.keras.Sequential([
            #tfkl.RandomBrightness(0.2, value_range=(0,1)),
            tfkl.RandomTranslation(0.15,0.15),
            #tfkl.RandomContrast(0.75),
            #tfkl.RandomBrightness(0.15),
            tfkl.RandomZoom(0.1),
            tfkl.RandomFlip("horizontal"),
            tfkl.RandomFlip("vertical"),
            tfkl.RandomRotation(0.2),
        ], name='preprocessing')

    augmentation = augmentation(inputs)

    x = externalNet(augmentation)

    # GAP layer only if pooling=None in ConvNeXtLarge parameters
    #x = tfkl.GlobalAveragePooling2D()(x)

    x = tfkl.Dropout(0.3)(x)

    x = tfkl.GlobalAveragePooling2D()(x)

    reg_strength = 0.03
    outputs = tfkl.Dense(
            2,
            kernel_regularizer=tfk.regularizers.l2(reg_strength),
            activation='softmax',
            name='Output'
        )(x)


    # Create a Model connecting input and output
    tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')

    # Compile the model with Categorical Cross-Entropy loss and Adam optimizer
    tl_model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=['accuracy'])

    # Display model summary
    tl_model.summary()
    # Train the model
    tl_history = tl_model.fit(
        x = images_train, # We need to apply the preprocessing thought for the MobileNetV2 network
        y = labels_train,
        batch_size = 32,
        epochs = 500,
        validation_data = (images_val, labels_val), # We need to apply the preprocessing thought for the MobileNetV2 network
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=100, restore_best_weights=True)]
    ).history

    # Save the best model
    tl_model.save('TransferLearningModel')
    del tl_model

    # Re-load the model after transfer learning
    ft_model = tfk.models.load_model('TransferLearningModel')
    ft_model.summary()

    # Set all ConvNeXtLarge layers as trainable
    ft_model.get_layer(network_keras_name).trainable = True
    #for i, layer in enumerate(ft_model.get_layer('mobilenetv2_1.00_96').layers):
    #    print(i, layer.name, layer.trainable)

    # Freeze first N layers, e.g., until the 125th one
    N = 125
    for i, layer in enumerate(ft_model.get_layer(network_keras_name).layers[:N]):
        layer.trainable=False
    for i, layer in enumerate(ft_model.get_layer(network_keras_name).layers):
        print(i, layer.name, layer.trainable)
    ft_model.summary()

    # Compile the model
    ft_model.compile(loss=tfk.losses.BinaryCrossentropy(), optimizer=tfk.optimizers.Adam(1e-5), metrics='accuracy')

    # Fine-tune the model
    ft_history = ft_model.fit(
        x = images_train,
        y = labels_train,
        batch_size = 32,
        epochs = 500,
        validation_data = (images_val, labels_val), 
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=100, restore_best_weights=True)]
    ).history

    # Save the model
    ft_model.save(model_name)

    # Set plot_result=True to plot accuracy and loss graphs
    plot_result = False
    if plot_result:
        plot_results(ft_history)

    evaluate = True
    if evaluate:
        # Evaluate the model on the test set

        # Predict labels for the entire test set
        predictions = ft_model.predict(images_test, verbose=0)

        # Display the shape of the predictions
        print("Predictions Shape:", predictions.shape)

        # Compute the confusion matrix
        cm = confusion_matrix(np.argmax(labels_test, axis=-1), np.argmax(predictions, axis=-1))

        # Compute classification metrics
        accuracy = accuracy_score(np.argmax(labels_test, axis=-1), np.argmax(predictions, axis=-1))
        precision = precision_score(np.argmax(labels_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
        recall = recall_score(np.argmax(labels_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
        f1 = f1_score(np.argmax(labels_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')

        # Display the computed metrics
        print('Accuracy:', accuracy.round(4))
        print('Precision:', precision.round(4))
        print('Recall:', recall.round(4))
        print('F1:', f1.round(4))

        print("\n0:Healthy, 1:Unhealthy\n")

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm.T, annot=True, xticklabels=np.unique(labels_test), yticklabels=np.unique(labels_test), cmap='Blues')
        plt.xlabel('True labels')
        plt.ylabel('Predicted labels')
        plt.show()

# ------------------------------------------
if __name__ == "__main__":
    train = True
    if (train):
        train_model()
    else:
        print("No training :(")

    print("Done!")