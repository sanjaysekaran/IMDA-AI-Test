import os
import cv2
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, optimizers, callbacks
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class Captcha:
    def __init__(self):
        self.model = None
        self.IMG_SIZE = (60, 30)  # (width, height)
        self.CHAR_WIDTH = self.IMG_SIZE[0] // 5
        self.CHAR_HEIGHT = self.IMG_SIZE[1]
        self.SEQ_LENGTH = 5
        self.CHARS = string.ascii_uppercase + string.digits
        self.CHAR_DICT = {char: idx for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS)

    def segment_characters(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.IMG_SIZE)  # Resize to known dimensions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply basic binary threshold (ignore background)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size
        char_regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 5 <= w <= 20 and 10 <= h <= 30:
                char_regions.append((x, y, w, h))

        # Sort left to right based on x coordinate
        char_regions = sorted(char_regions, key=lambda b: b[0])

        # Sanity check: merge or pad to get expected number of characters
        if len(char_regions) != self.SEQ_LENGTH:
            print(f"[WARNING] Found {len(char_regions)} character regions in {img_path}. Trying fallback slicing...")
            return self.fallback_fixed_slicing(image)

        char_images = []
        for x, y, w, h in char_regions:
            char = thresh[y:y+h, x:x+w]
            char = cv2.resize(char, (self.CHAR_WIDTH, self.CHAR_HEIGHT))  # Normalize to model input
            char_images.append(char / 255.0)

        return np.array(char_images)

    def fallback_fixed_slicing(self, image):
        """Fallback to fixed-width slicing if contours fail."""
        char_width = image.shape[1] // self.SEQ_LENGTH
        chars = []
        for i in range(self.SEQ_LENGTH):
            char_img = image[:, i*char_width:(i+1)*char_width]
            char_img = cv2.resize(char_img, (self.CHAR_WIDTH, self.CHAR_HEIGHT))
            chars.append(char_img / 255.0)
        return np.array(chars)

    def load_data(self, image_dir, label_dir):
        X, y = [], []
        input_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        output_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

        for input_fname, output_fname in zip(input_filenames, output_filenames):
            img_path = os.path.join(image_dir, input_fname)
            label_path = os.path.join(label_dir, output_fname)

            char_images = self.segment_characters(img_path)
            if len(char_images) != 5:
                continue  # Skip samples with incorrect segmentation

            with open(label_path, 'r') as f:
                label_text = f.read().strip()

            for i, char_img in enumerate(char_images):
                X.append(np.expand_dims(char_img, -1))  # Add channel dim
                y.append(self.CHAR_DICT[label_text[i]])

        X = np.expand_dims(np.array(X), -1)  # Add channel dimension
        y = np.array(y)

        return X, y

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.CHAR_HEIGHT, self.CHAR_WIDTH, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.NUM_CLASSES, activation='softmax')
        ])
  
        return model

    def load_model(self, model_path):
        print(f"Loading model from {model_path} ...")
        self.model = models.load_model(model_path)
        print("Model loaded successfully.")

    def __train__(self, train_input_path, train_output_path,
                  epochs=100, batch_size=4, learning_rate=0.0001,
                  validation_split=0.2, early_stopping_min_delta=0.01):

        print(
            f"Training with batch_size={batch_size}, "
            f"lr={learning_rate}, "
            f"validation_split={validation_split}, "
        )

        X, y = self.load_data(train_input_path, train_output_path)

        self.model = self.build_model()
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        early_stop = callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=early_stopping_min_delta, restore_best_weights=True)

        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            validation_split=validation_split,
            verbose=1
        )

        self.model.save("captcha_model.keras")
        print(f"Model saved to captcha_model.keras")

    def __call__(self, inference_input_path, inference_output_path):
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call __train__ or load_model first.")

        char_images = self.segment_characters(inference_input_path)
        if len(char_images) != 5:
            print(f"[ERROR] Failed to segment characters in {inference_input_path}")
            return

        predicted_text = ""
        for char_img in char_images:
            input_img = np.expand_dims(char_img, axis=(0, -1))
            pred = self.model.predict(input_img, verbose=0)
            predicted_text += self.CHARS[np.argmax(pred)]

        if not os.path.exists(os.path.dirname(inference_output_path)):
            os.makedirs(os.path.dirname(inference_output_path))
        with open(inference_output_path, 'w+') as f:
            f.write(predicted_text)

        print(f"Inference complete. Output: {predicted_text}")

# Example Usage
captcha = Captcha()

# Training
captcha.__train__(
    train_input_path="/train/input",       # .jpg images
    train_output_path="/train/output",      # .txt labels
    epochs=150,
    batch_size=4,
    learning_rate=0.0001,
    validation_split=0.0,
    early_stopping_min_delta=0.01
)

# If model is already trained, you can load model directly:
captcha.load_model("captcha_model.keras")

# Inference
captcha.__call__(
    inference_input_path="test/input/input100.jpg",
    inference_output_path="test/output/output100.txt"
)