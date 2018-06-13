#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 21:23:04 2018

@author: adam
"""
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
import numpy as np
import scipy.sparse as sp
import glob, os

#%%
imagespath= "/home/adam/artnetwork/fineartamericaspider/output/full"
os.chdir(imagespath)
images=glob.glob("*.jpg")
nimages=len(images)

model = VGG16(include_top=False, weights='imagenet')
n_dims=25088 #number of features extraccted from this model

X = np.zeros((nimages, 224, 224, 3))
#%%
#Start at 1 so there is room in the matrix to insert the user image at 0
#counter = 1
counter = 0

for imgname in images:
    x = kimage.load_img(imgname, target_size=(224, 224))
    x = kimage.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X[counter, :, :, :]=x
    counter+=1
    #%%
X = preprocess_input(X)
these_preds = model.predict(X)
#%%
np.save('/home/adam/artnetwork/saved_these_preds',these_preds)
#%%
shp = (nimages+1, n_dims)
preds=sp.lil_matrix((nimages+1, n_dims))
#preds[0:nimages+1, :] = these_preds.reshape(shp)
these_preds = these_preds.reshape(shp)
#%%
#%%
#preds = preds.tocsr()
np.save('/home/adam/artnetwork/savedfeatures',these_preds)
