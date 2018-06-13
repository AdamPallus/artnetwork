#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:52:42 2018

@author: adam
"""
#%%
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
from keras.applications import VGG16
import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm
from IPython.display import display, Image
import glob, os
import operator
import matplotlib.pyplot as plt

#%%
def cosine_distance(a,b):
    return(np.inner(a, b) / (norm(a) * norm(b)))

def euclidian_distance(a,b):
    return(norm(a-b)*-1)
    
def find_matches(imageurl,X,model,distance='cosine'): 
    img = kimage.load_img(imageurl, target_size=(224, 224))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    pred = pred.flatten()
    
    nimages = len(X)
    sims = np.zeros((nimages, 1))
    for i in range(0,nimages):
        if distance=='cosine':
            sims[i]= cosine_distance(pred.flatten(),X[i].flatten())
        else:
            sims[i]= euclidian_distance(pred.flatten(),X[i].flatten())
    return(sims)
    
def display_matches(imageurl,preds,model,images,nimages=10,distance='cosine'):
    display(Image(imageurl,height=300, width=300))
    similarities = find_matches(imageurl,preds,model,distance)
    similar_images=dict(zip(images,similarities))
    topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=True)[0:nimages]
    for match in topmatches:
        display(Image(match[0]))
        
def display_matches_inverse(imageurl,preds,model,images,nimages=10,distance='cosine'):
    display(Image(imageurl,height=300, width=300))
    similarities = find_matches(imageurl,preds,model,distance)
    similar_images=dict(zip(images,similarities))
    topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=False)[0:nimages]
    for match in topmatches:
        display(Image(match[0]))
        
#%%
'''
Load Features and Model
'''

collection_features = np.load('/home/adam/artnetwork/saved_collection_features.npy')


#%%
model = VGG16(include_top=False, weights='imagenet')

#%%
'''
Load images paths from disk
images are collected by the scrapy spider
'''
imagespath= "/home/adam/artnetwork/fineartamericaspider/output/full"
os.chdir(imagespath)
images=glob.glob("*.jpg")


#%%
chosenimage="/home/adam/Pictures/sun1.jpg"
display_matches(chosenimage,collection_features,model,images,nimages=4)

#%%
plt.hist(find_matches(chosenimage,collection_features,model),bins=100)
plt.show

#%%
from tkinter.filedialog import askopenfile
#%%
options ={}
options['initialdir'] = "/home/adam/Pictures"
#%%
chosenimage = askopenfile(**options).name
display_matches(chosenimage,collection_features,model,images,nimages=20,distance='cosine')

#%%
#%%
chosenimage = askopenfile(**options).name
display_matches(chosenimage,collection_features,model,images,nimages=50,distance='euclidian')

#%%
chosenimage = askopenfile(**options).name
display_matches_inverse(chosenimage,collection_features,model,images,nimages=50,distance='euclidian')
