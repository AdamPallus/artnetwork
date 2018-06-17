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
import pandas as pd
from scipy import sparse

#%%
fat=pd.read_csv('/home/adam/artnetwork/files_and_titles_trimmed.csv')
fat=fat
nimages=fat.shape[0]

model = VGG16(include_top=False, weights='imagenet')
n_dims=25088 #number of features extraccted from this model

#%%
#Start at 1 so there is room in the matrix to insert the user image at 0
#counter = 1
counter = 0
imagespath= "/home/adam/artnetwork/fineartamericaspider/output/full/"
features=[]
for imgname in fat.imgfile:
    x = kimage.load_img(imagespath+imgname, target_size=(224, 224))
    x = kimage.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = model.predict(x)
    x = sparse.csr_matrix(x.flatten())
    x[x > 0] = 1
    x = x.astype('int')
    features.append(x)
    
    #%%
sparsefeatures=[]
for feature in features:
    sparsefeatures.append()
    #%%
X = preprocess_input(X)
collection_features = model.predict(X)

#%%
np.save('/home/adam/artnetwork/collection_features2',collection_features)
#%%
shp = (nimages+1, n_dims)
collection_features = collection_features.reshape(shp)
#%%
features=sp.lil_matrix((nimages+1, n_dims))
preds[0:nimages+1, :] = these_preds.reshape(shp)
#%%
#%%
#preds = preds.tocsr()
np.save('/home/adam/artnetwork/savedfeatures',these_preds)
