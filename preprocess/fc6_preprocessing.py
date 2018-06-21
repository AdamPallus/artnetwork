#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New function to use the first fully connected layer (f6)
"""

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
import numpy as np
import glob, os
import pandas as pd

#imagespath= "/home/adam/artnetwork/fineartamericaspider/output/celeb"
imagespath= "/home/adam/artnetwork/fineartamericaspider/output/all"
os.chdir(imagespath)
images=sorted(glob.glob("*.jpg"))
nimages=len(images)

#include the top because we want the fully connected layer
model = VGG16(include_top=True, weights='imagenet')
#remove the classification layer (fc8)
model.layers.pop()
#remove the next fully connected layer (fc7)
model.layers.pop()
#fix the output of the model
model.outputs = [model.layers[-1].output]

n_dims=4096 #number of features extraccted from this model


fc6_features=[]

for imgname in images:
    x = kimage.load_img(imgname, target_size=(224, 224))
    x = kimage.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    fc6_features.append(model.predict(x))
#    
features = np.concatenate(fc6_features,axis=0)

np.save('fc6_features_all',fc6_features)
features[features>0]=1
np.save('fc6_features_all_binary',fc6_features)