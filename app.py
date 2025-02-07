from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load your models for cotton, brinjal, and tomato
cotton_leaf_model = load_model('models/model_resnet152V2.h5')
cotton_stem_model = load_model('models/StemModel.h5')
cotton_bud_model = load_model('models/BudModel.h5')
brinjal_model = load_model('models/BudModel.h5')  # Assuming similar model for brinjal
tomato_model = load_model('models/BudModel.h5')   # Assuming similar model for tomato

# Set up directories for each plant part
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels for diseases
leaf_class_labels = [
    "Aphids", "Army worm", "Bacterial Blight", "Leaf Curl Disease", 
    "Powdery Mildew", "Target Spot", "Fresh Cotton Leaf"
]

stem_class_labels = [
    "Southern Blight", "Fresh Stem", "Fusarium Wilt", "Verticillium Wilt"
]

bud_class_labels = [
    "Alteraria", "Boll Rot", "Botrytis Blight", "Cotton Pink BollWorm", "Fresh Bud"
]

brinjal_class_labels = [
    "Bacterial Wilt", "Fruit Rot", "Mite Infestation", "Healthy Brinjal"
]

tomato_class_labels = [
    "Early Blight", "Late Blight", "Septoria Leaf Spot", "Healthy Tomato"
]

# Mapping model prediction to respective part of the plant
def predict_part(plant, part, image_path):
    if plant == 'cotton':
        if part == 'leaf':
            model = cotton_leaf_model
            class_labels = leaf_class_labels
        elif part == 'stem':
            model = cotton_stem_model
            class_labels = stem_class_labels
        elif part == 'bud':
            model = cotton_bud_model
            class_labels = bud_class_labels
    elif plant == 'brinjal':
        model = brinjal_model
        class_labels = brinjal_class_labels
    elif plant == 'tomato':
        model = tomato_model
        class_labels = tomato_class_labels
    else:
        return None

    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = image_array.reshape((1, 224, 224, 3))
    prediction = model.predict(image_array)
    predicted_class = prediction.argmax(axis=1)[0]
    
    predicted_label = class_labels[predicted_class]
    return predicted_label


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/select', methods=['GET'])
def select():
    return render_template('select.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    plant = request.args.get('plant')
    part = request.args.get('part')

    # If part is None and plant is brinjal, allow upload for brinjal image
    if part is None and plant == 'brinjal':
        return render_template('upload.html', plant=plant, part='brinjal')

    # If part is None and plant is tomato, allow upload for tomato image
    if part is None and plant == 'tomato':
        return render_template('upload.html', plant=plant, part='tomato')

    # If part is None for other plants, redirect to a selection page
    if part is None:
        return redirect(url_for('select_part', plant=plant))

    # For all other cases, proceed with the normal upload page
    return render_template('upload.html', plant=plant, part=part)


@app.route('/predict', methods=['POST'])
def predict():
    plant = request.form.get('plant')
    part = request.form.get('part')
    file = request.files['image']

    if file and file.filename != '':
        # Save the image in the respective directory (uploads/plant/part)
        if part is None:  # No part specified for Brinjal/Tomato
            part = plant  # Consider the whole plant model
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], plant, part, file.filename)

        # Make sure the folder exists, if not, create it
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        # Save the file
        file.save(file_path)

        # Make prediction using the correct model based on plant and part
        predicted_label = predict_part(plant, part, file_path)

        # Return result page with predicted label and image
        return render_template('result.html', prediction=predicted_label, image_url=file_path)

    return "No file uploaded", 400


@app.route('/plant', methods=['GET', 'POST'])
def plant():
    return render_template('plant.html')


@app.route('/cotton', methods=['GET', 'POST'])
def cotton():
    return render_template('cotton.html')


@app.route('/brinjal', methods=['GET', 'POST'])
def brinjal():
    return redirect(url_for('upload', plant='brinjal', part=None))


@app.route('/tomato', methods=['GET', 'POST'])
def tomato():
    return redirect(url_for('upload', plant='tomato', part=None))


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
