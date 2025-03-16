from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("emnist_digit_model.h5")


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    if img is None:
        return None

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  
    img = cv2.resize(img, (28, 28))  

    img = img.astype("float32") / 255.0  
    img = np.expand_dims(img, axis=0)  
    img = np.expand_dims(img, axis=-1)  

    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = "uploaded_image.png"
    file.save(file_path)

    processed_img = preprocess_image(file_path)
    if processed_img is None:
        return jsonify({"error": "Could not process image"}), 400

    raw_predictions = model.predict(processed_img)
    predicted_digit = int(np.argmax(raw_predictions))

    os.remove(file_path)  # Clean up

    return jsonify({"prediction": predicted_digit})


if __name__ == "__main__":
    app.run(debug=True)
