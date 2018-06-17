#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main script for controlling the website behavior
We load the saved features of the images and links to those
images on the site fineartamerica.com

Then we find the best matches and send those to results.html

@author: adam
"""

from app import app
from app.models import similarity
import flask

import glob, os, io
from scipy import sparse
import numpy as np
from keras.applications import VGG16
#from keras.applications import mobilenet
from PIL import Image
from keras.preprocessing import image as kimage
import tensorflow as tf
from werkzeug.utils import secure_filename
from scipy.spatial.distance import cosine

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

from settings import APP_STATIC
collection_data = pd.read_pickle('/home/adam/artnetwork/mvpapp/app/static/artcollection.pickle')
#collection_data = pd.read_pickle(os.path.join(APP_STATIC,'artcollection.pickle'))

imagespath= "/home/adam/artnetwork/fineartamericaspider/output/full"

app.secret_key = 'adam'

os.chdir(imagespath)
images=glob.glob("*.jpg")

graph = tf.get_default_graph()
model = VGG16(include_top=False, weights='imagenet')
#model = mobilenet.MobileNet(include_top=False,weights='imagenet')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def hamming_distance(a,b):
    '''
    compares distance for binary arrays
    returns number of features that are not the same
    '''
    c=(a+b)>1
    return(c.sum())
    
def find_matches(pred, collection_data): 
    nimages=collection_data['imgfile'].size
    sims = np.zeros((nimages,1))
    for i in range(0, nimages):
        sims[i]=cosine_similarity(pred, collection_data['features'][i])
    return(sims.flatten())

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
            pred = sparse.csr_matrix(pred.flatten())
            pred[pred > 0] = 1
            pred = pred.astype('int')
            collection_data['simscore']=find_matches(pred, collection_data)
            showresults=collection_data.drop(columns='features')
            showresults=showresults.sort_values(by='simscore',ascending=False)
            showresults=showresults.reset_index()
            # Delete image when done with analysis
#            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            return flask.render_template('results2.html',matches=showresults,original=img_name)
        flask.flash('Upload only image files')

        
        return flask.redirect(flask.request.url)
    