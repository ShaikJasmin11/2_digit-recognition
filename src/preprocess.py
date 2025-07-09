import numpy as np
from tensorflow.keras.datasets import mnist

def load_and_prepare_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize to 0-1
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Reshape for CNN: (28,28) => (28,28,1)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    return X_train, y_train, X_test, y_test
