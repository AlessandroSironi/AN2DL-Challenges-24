import os
import tensorflow as tf

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'CNN_5_GAP.keras'))

    def predict(self, X):
        
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        out = self.model.predict(X)
        #out = tf.argmax(out, axis=-1)  # Shape [BS]
        out = tf.convert_to_tensor((out > 0.5).astype(int).flatten())

        return out
