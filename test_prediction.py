import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt


model = load_model("emnist_digit_model.h5")


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    if img is None:
        raise ValueError(f"Error: Image at {image_path} not found or unreadable!")

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  
    img = cv2.resize(img, (28, 28))

    img = img.astype("float32") / 255.0  
    img = np.expand_dims(img, axis=0)  
    img = np.expand_dims(img, axis=-1)  

    return img


def softmax_with_temp(logits, temp=1.5):
    """Apply temperature scaling to soften predictions"""
    exp_logits = np.exp(logits / temp)
    return exp_logits / np.sum(exp_logits)


image_path = "9.png"  
processed_img = preprocess_image(image_path)


raw_predictions = model.predict(processed_img)

# Apply temperature scaling to reduce overconfidence
soft_predictions = softmax_with_temp(raw_predictions)

# Get predicted digit
predicted_digit = np.argmax(soft_predictions)

# Display results
print(f" Raw Model Output: {raw_predictions}")
print(f" Softmax Scaled Output: {soft_predictions}")
print(f" Predicted Digit: {predicted_digit}")

# Show the processed image
plt.imshow(processed_img.reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {predicted_digit}")
plt.show()
