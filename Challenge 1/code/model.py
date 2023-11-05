import os

#Fix randomness and hide warnings
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

# Import tensorflow
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
print(tf.__version__)

# Import other libraries
#import cv2
#from tensorflow.keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

print("Finished loading libraries")

model_name = "CNN_1"

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, model_name))

    def predict(self, X):
        
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)  # Shape [BS]

        return out
    

def show_images(images):
    # Show all images in in images array. Make a scrollable window if there are more than 50 images and display them in a grid
    num_images = len(images)    # Number of images
    num_cols = int(np.ceil(np.sqrt(num_images)))    # Number of columns in the grid
    num_rows = int(np.ceil(num_images / num_cols))

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()
    
def train_model():
    # Conditional check for unzipping
    """ unzip = False

    if unzip:
        import zipfile
        # Unzip the public_data.zip file if the 'unzip' flag is True
        with zipfile.ZipFile('../data/phase_1/public_data.zip', 'r') as zip_ref:
            zip_ref.extractall('data/phase_1/') """

    # Load images from the .npz file
    data_path = 'public_data.npz'
    data = np.load(data_path, allow_pickle=True)

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

    # ------------------------------------------
    # Display images to check if correctly loaded
    
    display_images = False
    if display_images:
        show_images(images)

    # ------------------------------------------
    labels = np.array(labels) #TODO: Check if needed

    labels = LabelEncoder().fit_transform(labels)
    labels = tfk.utils.to_categorical(labels,len(np.unique(labels)))

    # Use the stratify option to maintain the class distribution in the train and test datasets
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, stratify=np.argmax(labels, axis=1), random_state=seed)

    # Further split the test set into test and validation sets, stratifying the labels
    images_test, images_val, labels_test, labels_val = train_test_split(images_test, labels_test, test_size=0.5, stratify=np.argmax(labels_test, axis=1), random_state=seed)

    print("\n\nSHAPES OF THE SETS:\n")

    print(f"images_train shape: {images_train.shape}, labels_train shape: {labels_train.shape}")
    print(f"images_val shape: {images_val.shape}, labels_val shape: {labels_val.shape}")
    print(f"images_test shape: {images_test.shape}, labels_test shape: {labels_test.shape}")

    print("\n\n")

    # ------------------------------------------
    # Define input shape, output shape, batch size, and number of epochs
    input_shape = images_train.shape[1:]
    output_shape = labels_train.shape[1:]
    batch_size = 32
    epochs = 1000

    # Print input shape, batch size, and number of epochs
    #print(f"Input Shape: {input_shape}, Output Shape: {output_shape}, Batch Size: {batch_size}, Epochs: {epochs}")
    # ------------------------------------------

    callbacks = [
        tfk.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True, mode='auto'),
    ]

    # INSERT AUGMENTATION HERE
    def build_model(input_shape=input_shape, output_shape=output_shape):
        tf.random.set_seed(seed)

        # Build the neural network layer by layer
        input_layer = tfkl.Input(shape=input_shape, name='Input')

        x = tfkl.Conv2D(filters=32, kernel_size=3, padding='same', name='conv0')(input_layer)
        x = tfkl.ReLU(name='relu0')(x)
        x = tfkl.MaxPooling2D(name='mp0')(x)

        x = tfkl.Conv2D(filters=64, kernel_size=3, padding='same', name='conv1')(x)
        x = tfkl.ReLU(name='relu1')(x)
        x = tfkl.MaxPooling2D(name='mp1')(x)

        x = tfkl.Conv2D(filters=128, kernel_size=3, padding='same', name='conv2')(x)
        x = tfkl.ReLU(name='relu2')(x)
        x = tfkl.MaxPooling2D(name='mp2')(x)

        x = tfkl.Conv2D(filters=256, kernel_size=3, padding='same', name='conv3')(x)
        x = tfkl.ReLU(name='relu3')(x)
        x = tfkl.MaxPooling2D(name='mp3')(x)

        x = tfkl.Conv2D(filters=512, kernel_size=3, padding='same', name='conv4')(x)
        x = tfkl.ReLU(name='relu4')(x)

        x = tfkl.GlobalAveragePooling2D(name='gap')(x)

        output_layer = tfkl.Dense(units=2, activation='softmax',name='Output')(x)

        # Connect input and output through the Model class
        model = tfk.Model(inputs=input_layer, outputs=output_layer, name='CNN')

        # Compile the model
        model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=['accuracy'])

        # Return the model
        return model

    model = build_model()
    model.summary()
    #tfk.utils.plot_model(model, expand_nested=True, show_shapes=True)

    # Train the model
    history = model.fit(
        x = images_train,
        y = labels_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (images_val, labels_val),
        callbacks = callbacks
    ).history

    model.save(model_name)

    """ # Plot the training
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

    plt.show() """

    # ------------------------------------------
if __name__ == "__main__":
    train = True
    if (train):
        train_model()
    #else:
        #print("No training :(")
    #_model = model(os.getcwd())


    """ # Load images from the .npz file
    __data_path = 'Challenge 1/data/phase_1/public_data.npz'
    __data = np.load(__data_path, allow_pickle=True)

    __images = __data['data']
    __labels = __data['labels']

    i = 0
    for __image in __images: 
        # Normalize image pixel values to a float range [0, 1]
        __images[i] = (__images[i] / 255).astype(np.float32)
        # Convert image from BGR to RGB
        __images[i] = __images[i][...,::-1]
        if (i % 100 == 0):
            print("Processing image: ", i, "\n")
            #pred = _model.predict(__image)
            #print("Prediction: ", pred, " - Label: ", __labels[i])
        i = i+1

    # print the shape of __images
    print(__images.shape)

    pred = _model.predict(__images)
    print(pred)

    for y in pred:
        print(y, "\n") """

    #print("Done!")