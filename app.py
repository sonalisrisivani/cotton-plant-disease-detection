from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'static/model_resnet152V2.h5'
model = load_model(MODEL_PATH)

# Define the disease classes
CLASSES = [
    'Aphids', 'Army Worm', 'Bacterial Blight',
    'Fresh Cotton Leaf', 'Leaf Curl Disease',
    'Powdery Mildew', 'Target Spot'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected.", 400

    if file:
        # Save the uploaded image to static folder
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Load the image for prediction
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        predicted_class = CLASSES[class_index]

        return render_template(
            'result.html', prediction=predicted_class, image_path=file_path
        )

if __name__ == '__main__':
    app.run(debug=True)
