# Cotton Plant Disease Detection

This repository contains a Flask web application that predicts diseases in cotton plants based on uploaded images using a pre-trained deep learning model. The model identifies the disease from a set of predefined classes and provides the corresponding disease name.

## Features

- Upload cotton plant images for disease prediction.
- View the result with the predicted disease class and the uploaded image.
- Displays the prediction on a web page after processing the image.

## Requirements

To run the project, you need to have the following installed:

- Python 3.x
- Flask
- TensorFlow (for loading the `.h5` model)
- Keras
- PIL (Pillow for image handling)

You can install the necessary Python packages using `pip`:

```bash
pip install -r requirements.txt


## Setup

1. Clone the repository and navigate to the project directory:

```bash
git clone "https://github.com/sonalisrisivani/cotton-plant-disease-detection"
cd cotton-plant-disease-detection

2. Install Virtual Environment Packages
```bash
python -m venv venv 

3. Activate Virtual Environment
```bash
venv\Scripts\activate # For Windows

4. Install dependencies 
```bash
pip install -r requirements.txt

5. Run application 
```bash
python app.py 
or
```bash 
flask run 

Go to - localhost/5000

6. deactivate
```bash 
deactivate


## License
This project is licensed under the MIT License - see the LICENSE file for details.
