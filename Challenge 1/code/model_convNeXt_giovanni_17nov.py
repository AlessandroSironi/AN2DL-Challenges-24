
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
from tensorflow.keras.applications.mobilenet import preprocess_input
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
#print(tf.__version__)

# Import other libraries
#import cv2
#from tensorflow.keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

print("Finished loading libraries")

model_name = "model_convNeXt_giovanni_17nov"

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

def plot_results(history):
    """ print("Plotting results...")
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

    plt.show() """

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


def train_model():
    print("[*] Training model ", model_name, "...")
    # Load images from the .npz file
    data_path = 'public_data.npz'
    data = np.load(data_path, allow_pickle=True)

    images = data['data']
    labels = data['labels']

    """ i = 0
    for image in images:
        # Normalize image pixel values to a float range [0, 1]
        #images[i] = (images[i] / 255).astype(np.float32)

        # Convert image from BGR to RGB
        #images[i] = images[i][...,::-1]
        i = i+1
        if (i % 1000 == 0):
            print("Processing image: ", i)
    print("Finished processing images") """
    # EfficientNetV2 models expect their inputs to be float tensors of pixels with values in the [0-255] range.
    images = (images).astype(np.float32)

    # ------------------------------------------
    # Sanitize input
    # Delete trolololol and shrek
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
    def mse(imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err
    positions_to_remove = []

    pos_shrek = 58
    pos_trolo = 338
    for pos in range(len(images)):
        if (mse(images[pos_shrek],images[pos])==0 or mse(images[pos_trolo],images[pos])==0):
            positions_to_remove.append(pos)
    if (positions_to_remove != positions_to_remove_old):
        print("ERROR: Different positions to remove")
        exit()
    print("Len of positions_to_remove: ", len(positions_to_remove))
    n = 0

    for pos in positions_to_remove:
        new_pos = pos - n
        #print("Removing image at position: ", pos, " - New Position is ", new_pos)
        images = np.delete(images, new_pos, axis=0)
        labels = np.delete(labels, new_pos, axis=0)
        n = n + 1

    # ------------------------------------------

    labels = np.array(labels)

    labels = LabelEncoder().fit_transform(labels)
    labels = tfk.utils.to_categorical(labels,len(np.unique(labels)))

    # Use the stratify option to maintain the class distribution in the train and test datasets
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.15, stratify=np.argmax(labels, axis=1), random_state=seed)

    # Further split the test set into test and validation sets, stratifying the labels
    images_test, images_val, labels_test, labels_val = train_test_split(images_test, labels_test, test_size=0.9, stratify=np.argmax(labels_test, axis=1), random_state=seed)

    print("\n\nSHAPES OF THE SETS:\n")

    print(f"images_train shape: {images_train.shape}, labels_train shape: {labels_train.shape}")
    print(f"images_val shape: {images_val.shape}, labels_val shape: {labels_val.shape}")
    print(f"images_test shape: {images_test.shape}, labels_test shape: {labels_test.shape}")

    print("\n\n")

    input_shape = images_train.shape[1:]
    output_shape = labels_train.shape[1:]

    # ------------------------------------------
    #Print input shape, batch size, and number of epochs
    #print(f"Input Shape: {input_shape}, Output Shape: {output_shape}, Batch Size: {batch_size}, Epochs: {epochs}")
    # ------------------------------------------
    #if include_preprocessing=True no preprocessing is needed
    """ efficientNet = tf.keras.applications.EfficientNetV2L(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
        include_preprocessing=True,
    ) """

    externalNet = tf.keras.applications.ConvNeXtLarge(
        model_name="convnext_large",
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        #input_tensor=None,
        input_shape=input_shape,
        pooling="avg",
        #classes=1000,
        #classifier_activation="softmax",
    )

    #Automatically get the name of the network
    network_keras_name = externalNet.name
    print("[*] Network name: ", network_keras_name)

    """mobile = tf.keras.applications.MobileNetV3Large(
        input_shape=None,
        alpha=1.0,
        minimalistic=False,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        include_preprocessing=True,
    )"""

    for i, layer in enumerate(externalNet.layers):
        print(i, layer.name, layer.trainable)

    #tfk.utils.plot_model(mobile, show_shapes=True)
    # Use the supernet as feature extractor, i.e. freeze all its weigths
    externalNet.trainable = False

    # Create an input layer with shape (224, 224, 3)
    inputs = tfk.Input(shape=(96, 96, 3))

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

    #not needed
    #scale_layer = tfkl.Rescaling(scale = 1/127.5, offset = -1)
    #x = scale_layer(augmentation)

    #x = mobile(augmentation)
    x = externalNet(augmentation)

    """ x = tfkl.Conv2D (
        filters = 128,
        kernel_size = (3,3),
        activation = 'relu',
        name = 'Conv2D_1'
    ) (x) """

    #x = tfkl.GlobalAveragePooling2D()(x)

    x = tfkl.Dropout(0.5)(x)

    reg_strength = 0.04
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
        epochs = 200,
        validation_data = (images_val, labels_val), # We need to apply the preprocessing thought for the MobileNetV2 network
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=20, restore_best_weights=True)]
    ).history

    # Save the best model
    tl_model.save('TransferLearningModel')
    del tl_model

    # Re-load the model after transfer learning
    ft_model = tfk.models.load_model('TransferLearningModel')
    ft_model.summary()

    # Set all MobileNetV2 layers as trainable
    ft_model.get_layer(network_keras_name).trainable = True
    #for i, layer in enumerate(ft_model.get_layer('mobilenetv2_1.00_96').layers):
    #    print(i, layer.name, layer.trainable)

    # Freeze first N layers, e.g., until the 133rd one
    #N = 270
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
        x = images_train, # We need to apply the preprocessing thought for the MobileNetV2 network
        y = labels_train,
        batch_size = 32,
        epochs = 50,
        validation_data = (images_val, labels_val), # We need to apply the preprocessing thought for the MobileNetV2 network
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=15, restore_best_weights=True)]
    ).history

    # Save the model
    ft_model.save(model_name)

    # ------------------------------------------
    plot_result = True
    if plot_result:
        plot_results(ft_history)
    # ------------------------------------------

    evaluate = True
    if evaluate:
        # Evaluate the model on the test set
        # Copilot generated version:
        # test_loss, test_acc = model.evaluate(images_test, labels_test, verbose=2)

        # Notebooks version:
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

        #print("\n0:Healthy, 1:Unhealthy\n")
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

    print("Done!")