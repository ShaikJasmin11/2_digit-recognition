# main.py

from src.preprocess import load_and_prepare_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save():
    X_train, y_train, X_test, y_test = load_and_prepare_data()

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    model = build_cnn_model()
    model.fit(X_train, y_train_cat, epochs=5, validation_data=(X_test, y_test_cat))
    model.save('models/cnn_model.h5')
    print("✅ CNN model saved as cnn_model.h5")

if __name__ == "__main__":
    train_and_save()
