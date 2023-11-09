import os
import tensorflow as tf
import numpy as np

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, "CNN_5_GAP.keras"))

    def binary(self, n):
        if n > 0.5:
            n = 1
        else:
            n = 0
        return n

    def predict(self, X):
        
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        X = X / 255
        out = self.model.predict(X)
        out = np.array(list(map(self.binary, out)))
        out = tf.convert_to_tensor(out)  # Shape [BS]
        return out