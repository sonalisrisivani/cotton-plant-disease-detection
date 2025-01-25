from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('models', 'model_resnet152V2.h5')
model = load_model(model_path)

# Define the target image size (e.g., 224x224 for ResNet152V2)
IMAGE_SIZE = (224, 224)

# Route for the homepage (image upload form)
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the uploaded image and displaying the result
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    # Get the uploaded image
    file = request.files['image']

    if file.filename == '':
        return "No file selected", 400

    # Save the image temporarily
    image_path = os.path.join('static', 'uploads', file.filename)
    file.save(image_path)

    # Preprocess the image
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image_array = img_to_array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index

    # Optionally, map the predicted class index to a class label
    # e.g., class_labels = ['Class1', 'Class2', 'Class3']
    # predicted_label = class_labels[predicted_class]

    class_labels = [
        "Aphids",  # Class 0
        "Army worm",  # Class 1
        "Bacterial Blight",  # Class 2
        #"Fresh Cotton Leaf",  # Class 3
        "Leaf Curl Disease",  # Class 4
        "Powdery Mildew",  # Class 5
        "Target Spot",  # Class 6
        "Fresh Leaf"
        # Add other disease names as needed
    ]


    #predicted_label = f"Class {predicted_class}"
    # Get the predicted disease name
    predicted_label = class_labels[predicted_class]


    # Render the result.html page with the prediction and image
    return render_template('result.html', prediction=predicted_label, image_url=image_path)

if __name__ == '__main__':
    # Ensure the 'static/uploads' directory exists
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    app.run(debug=True)
