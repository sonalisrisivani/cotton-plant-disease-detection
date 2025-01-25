from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load your models for leaf, stem, and bud
leaf_model = load_model('models/model_resnet152V2.h5')
stem_model = load_model('models/StemModel.h5')
bud_model = load_model('models/model_resnet152V2.h5')

# Set up directories for each plant part
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Mapping model prediction to respective part of the plant
def predict_part(part, image_path):
    if part == 'leaf':
        model = leaf_model
    elif part == 'stem':
        model = stem_model
    elif part == 'bud':
        model = bud_model
    else:
        return None

    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = image_array.reshape((1, 224, 224, 3))
    prediction = model.predict(image_array)
    predicted_class = prediction.argmax(axis=1)[0]

    return predicted_class


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/select', methods=['GET'])
def select():
    return render_template('select.html')


@app.route('/upload', methods=['GET'])
def upload():
    part = request.args.get('part')
    return render_template('upload.html', part=part)


@app.route('/predict', methods=['POST'])
def predict():
    part = request.form.get('part')
    file = request.files['image']

    if file and file.filename != '':
        # Save the image in the respective directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], part, file.filename)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        file.save(file_path)

        # Make prediction using the respective model
        predicted_class = predict_part(part, file_path)

        # Optionally, map the predicted class index to a disease name
        # For now, we just print the class number (you can modify as per your needs)
        return render_template('result.html', prediction=f"Class {predicted_class}", image_url=file_path)

    return "No file uploaded", 400


if __name__ == '__main__':
    app.run(debug=True)
