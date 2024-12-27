import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('rock_classification_model.h5')

# Define the categories of rocks
rock_types = ['Igneous', 'Metamorphic', 'Sedimentary']


# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        img_file = request.files['image']

        # Save the image
        img_path = os.path.join('uploads', img_file.filename)
        img_file.save(img_path)

        # Prepare image for prediction
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        # Get the rock type name
        result = rock_types[predicted_class[0]]
        return render_template('result.html', result=result)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
