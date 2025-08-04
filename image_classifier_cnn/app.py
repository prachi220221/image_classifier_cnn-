
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model/model.h5')
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        image = request.files['image']
        img = Image.open(image).resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1, 32, 32, 3))
        predictions = model.predict(img_array)
        predicted_class = classes[np.argmax(predictions)]
        prediction = f'Predicted Class: {predicted_class}'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
