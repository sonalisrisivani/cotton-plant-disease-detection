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

# Class labels for Leaf, Stem, and Bud diseases
# Class labels for Leaf diseases
leaf_class_labels = [
    "Aphids",  # Class 0
    "Army worm",  # Class 1
    "Bacterial Blight",  # Class 2
      # Class 3
    "Leaf Curl Disease",  # Class 4
    "Powdery Mildew",  # Class 5
    "Target Spot",  # Class 6
    "Fresh Cotton Leaf"
    # Add other leaf disease names as needed
]


stem_class_labels = [
     "Southeren Blight" ,"Fresh Stem" ,  "Fusarium Wilt" , "Verticillium Wilt" 
]

bud_class_labels = [
    "Alteraria", "Boll Rot", "Botrytis Blight", "Cotton Pink BollWorm", "Fresh Bud"
]

# Mapping model prediction to respective part of the plant
def predict_part(part, image_path):
    if part == 'leaf':
        model = leaf_model
        class_labels = leaf_class_labels
    elif part == 'stem':
        model = stem_model
        class_labels = stem_class_labels
    elif part == 'bud':
        model = bud_model
        class_labels = bud_class_labels
    else:
        return None

    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = image_array.reshape((1, 224, 224, 3))
    prediction = model.predict(image_array)
    predicted_class = prediction.argmax(axis=1)[0]
    
    # Get the predicted disease name
    predicted_label = class_labels[predicted_class]
    return predicted_label


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
        predicted_label = predict_part(part, file_path)

        # Return result page with predicted label and image
        return render_template('result.html', prediction=predicted_label, image_url=file_path)

    return "No file uploaded", 400


if __name__ == '__main__':
    app.run(debug=True)
