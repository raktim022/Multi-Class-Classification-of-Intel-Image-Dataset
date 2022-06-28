import numpy as np
from flask import Flask, request, render_template,url_for,redirect
import pickle
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import keras

PATH = os.path.join("C:/", "Users/", "rakti/", "Downloads/", "vgg_multi_classifier.h5")
model = load_model(PATH)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('img_class.html')


@app.route('/login', methods=['POST'])
def predict():
    labels = ['Building', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
    img=request.files['fl']
    img_path=os.path.join("C:/", "Users/", "rakti/", "Downloads/", img.filename)
    img.save(img_path)
    img_pred=keras.utils.load_img(img_path,target_size=(256,256))
    img_pred=keras.utils.img_to_array(img_pred)
    img_pred=np.expand_dims(img_pred, axis=0)
    rslt= model.predict(img_pred)
    outp=labels[np.argmax(rslt)]
    outp1=f"The above image is of a {outp} with probability {np.round(max(rslt[0]),2)}"
    return render_template('img_class.html', pred=outp1)



if __name__ == '__main__':
    app.run(debug=True)