import numpy as np
import cv2
from keras import models

# Load the pre-trained model
model = models.load_model('model.h5')  # Ensure the model is in the same directory

# Define the characters to recognize
characters = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]  # 0-9 and A-Z

# Create a blank white image (28x28)
img = np.ones((28, 28), dtype=np.uint8) * 255  # White background

# Draw a simple digit (e.g., 7)
img[5:10, 10:20] = 0  # Draw a horizontal line (black)
img[10:20, 15:20] = 0  # Draw a vertical line (black)

# Display the image (optional)
cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Preprocess the image (same as in your Flask app)
img = img.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)
img = img.astype('float32') / 255.0  # Normalize pixel values

# Make a prediction
prediction = model.predict(img)
predicted_char = characters[np.argmax(prediction)]

# Print the result
print("Predicted character:", predicted_char)