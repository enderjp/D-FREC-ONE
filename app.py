# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:40:21 2021

@author: Ender
"""
from flask import Flask
from flask import request, render_template
import numpy as np
import pandas as pd
from copy import deepcopy
from werkzeug.utils import secure_filename
from PIL import Image
import dlib_align
import cv2
import clasificar
from clasificar import cnn
import shutil
# Importar clasificador - CNN
from tensorflow import keras
from tensorflow.keras import layers
import os

app = Flask(__name__, template_folder='templates')

# @app.route("/")
# def hello_world():
#     return "Hello World!!"

model = keras.models.load_model('model/clasificador_cnn2.h5')

@app.route('/',methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        
        shutil.rmtree('static/images/') 
        os.mkdir('static/images/')
        return( render_template('main.html'))
    
       # shutil.rmtree('static/images/')
       
        
        
    if  request.method == 'POST':
        f = request.files['file']
    
        file_name = 'static/images/'+str(secure_filename(f.filename))
        file_name2='images/'+str(secure_filename(f.filename))
        f.save(file_name)
        # realizar clasificación con la red neuronal
        cnn(file_name,model)
        
        
        #clf.save_image(f.filename)
        
       # return render_template('result.html',genero=genero,raza=raza, edad=edad, file_name2 = file_name)
        return render_template('result2.html',file_name = file_name2)

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0')




