#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:56:53 2018

@author: adam
"""

from app import app
from app.models import similarity
import flask

import glob, os, io

import numpy as np
from keras.applications import VGG16
from PIL import Image
from keras.preprocessing import image as kimage
import tensorflow as tf
from werkzeug.utils import secure_filename

import pandas as pd

collection_features = np.load('/home/adam/artnetwork/saved_collection_features.npy')

files_and_titles=pd.read_csv('/home/adam/Downloads/files_and_titles.csv')

imagespath= "/home/adam/artnetwork/fineartamericaspider/output/full"

app.secret_key = 'adam'

os.chdir(imagespath)
images=glob.glob("*.jpg")

graph = tf.get_default_graph()
model = VGG16(include_top=False, weights='imagenet')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/',  methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
        # Get method type
    method = flask.request.method
    print(method)


    if method == 'GET':
        return flask.render_template('index.html')
    
    if method == 'POST':
        # No file found in the POST submission
        if 'file' not in flask.request.files:
            print("FAIL")
            return flask.redirect(flask.request.url)

        # File was found
        file = flask.request.files['file']
        if file and allowed_file(file.filename):
            print('SUCCESS')
                    # Image info
            img_file = flask.request.files.get('file')
#            img_name = img_file.filename
            img_name = secure_filename(img_file.filename)
            # Write image to static directory and do the hot dog check
            imgurl=os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            img_file.save(imgurl)
            img = kimage.load_img(imgurl, target_size=(224, 224))
            img = kimage.img_to_array(img)
            img = np.expand_dims(img, axis=0)    
            global graph
            with graph.as_default():
                pred=model.predict(img)
            matches=similarity.find_matches(pred, collection_features, images,nimages=50)
            matches = pd.DataFrame(matches, columns=['imgfile', 'simscore'])
            matches['simscore']=matches.simscore.astype('double')
            showresults=files_and_titles.set_index('imgfile',drop=False).join(matches.set_index('imgfile'))
            showresults=showresults.sort_values(by='simscore',ascending=False)
            # Delete image when done with analysis
#            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            return flask.render_template('results.html',matches=showresults,original=img_name)
        flask.flash('Upload only image files')

        
        return flask.redirect(flask.request.url)
    