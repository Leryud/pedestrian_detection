import os
from flask import Flask, redirect, url_for, request, render_template
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

app = Flask(__name__)
os.path.dirname(__file__)

@app.route("/project/public/")
def index():
    return render_template("index.html")

@app.route('/public/result/', methods = ["POST"])
def prediction_result():
    uploaded_file = request.files['image']
    file_name = uploaded_file.filename
    UPLOAD_FOLDER = os.path.abspath('static/')
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(UPLOAD_FOLDER, file_name))

    IMAGE_SIZE = 224
    loadmodel = tf.keras.models.load_model(pedestrian.model)
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size = (IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)

    predictions = loadmodel.predict(img_array, steps=1)
    int_result = np.argmax(prediciton[0])

    if(int_result ==0):
        decision = 'no presence of pedestrians'

    else: 
        decision = 'presence of pedestrian'
    
    conf = 100*np.argmax(predictions[0])

    return render_template('result.html', status = decision, confidence = conf, upload_name = file_name)


if __name__ == '__main__':
    app.run()
