#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:52:42 2018

@author: adam
"""
#%%
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm
#%%
X = np.load('/home/adam/artnetwork/savedfeatures.npy')
X = X.all()
X = X.toarray()

#%%
from keras.applications import VGG16
model = VGG16(include_top=False, weights='imagenet')

#%%
def cosine_distance(a,b):
    return(np.inner(a, b) / (norm(a) * norm(b)))
    
def find_matches(imageurl,X,model): 
    img = kimage.load_img(imageurl, target_size=(224, 224))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    pred = pred.flatten()
    
    nimages = len(X)
    sims = np.zeros((nimages, 1))
    for i in range(0,nimages):
        sims[i]= cosine_distance(pred.flatten(),X[i].flatten())
    return(sims)

#%%
#%%
from IPython.display import display, Image
import glob, os
import operator
#%%
imagespath= "/home/adam/artnetwork/fineartamericaspider/output/full"
os.chdir(imagespath)
images=glob.glob("*.jpg")

#%%
topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=True)[0:10]
for match in topmatches:
    display(Image(match[0]))
    
#%%
def display_matches(imageurl,preds,model,images,nimages=10):
    display(Image(imageurl))
    similarities = find_matches(imageurl,preds,model)
    similar_images=dict(zip(images,similarities))
    topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=True)[0:nimages]
    for match in topmatches:
        display(Image(match[0]))

#%%
chosenimage="/home/adam/Pictures/cpl.jpeg"
display_matches(chosenimage,these_preds,model,images,nimages=15)

#%%
import matplotlib.pyplot as plt
plt.hist(find_matches(chosenimage,these_preds,model),bins=100)
plt.show