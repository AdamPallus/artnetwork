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
    
def get_preds(imageurl,model): 
    img = kimage.load_img(imageurl, target_size=(224, 224))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    pred = pred.flatten()  
    return(pred)

def find_matches_preds(pred,X,distance='cosine'):  
    nimages = len(X)
    sims = np.zeros((nimages, 1))
    for i in range(0,nimages):
        if distance=='cosine':
            sims[i]= cosine_distance(pred.flatten(),X[i].flatten())
        else:
            sims[i]= euclidian_distance(pred.flatten(),X[i].flatten())
    return(sims)
    
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

#    return(similar_images)
        
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
def match_two(imageurl1,imageurl2,collection_features,model,images,nimages=10,distance='cosine'):
    display(Image(imageurl1,height=300, width=300))
    display(Image(imageurl2,height=300, width=300))
    preds1=get_preds(imageurl1,model)
    preds2=get_preds(imageurl2,model)
    preds=(preds1+preds2)/2
    similarities = find_matches_preds(preds,collection_features,distance)
    similar_images=dict(zip(images,similarities))
    topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=False)[0:nimages]
    for match in topmatches:
        display(Image(match[0]))

def display_matches_pair(imageurl,imageurl2,collection_features,model,images,nimages=10,distance='cosine'):
    display(Image(imageurl,height=300, width=300))
    display(Image(imageurl2,height=300, width=300))
    similarities = find_matches(imageurl,collection_features,model,distance)
    similarities2 = find_matches(imageurl2,collection_features,model,distance)
    similar_images=dict(zip(images,(similarities+similarities2)/2))
    topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=True)[0:nimages]
    for match in topmatches:
        display(Image(match[0]))
#%%
chosenimage = askopenfile(**options).name
chosenimage2= askopenfile(**options).name
#%%
display_matches_pair(chosenimage,chosenimage,collection_features,model,images,nimages=15)
#%%
display_matches_pair(chosenimage2,chosenimage2,collection_features,model,images,nimages=15)
#%%
display_matches_pair(chosenimage,chosenimage2,collection_features,model,images,nimages=15)
#%%
match_two(chosenimage,chosenimage2,collection_features,model,images,nimages=20,distance='cosine')
#%%
match_two(chosenimage,chosenimage,collection_features,model,images,nimages=20,distance='cosine')
#%%
chosenimage = askopenfile(**options).name
display_matches(chosenimage,collection_features,model,images,nimages=50,distance='euclidian')

#%%
chosenimage = askopenfile(**options).name
display_matches_inverse(chosenimage,collection_features,model,images,nimages=50,distance='euclidian')
#%%
#messing with panda
import pandas as pd
#%%
files_and_titles=pd.read_csv('/home/adam/Downloads/files_and_titles.csv')
#%%
def cosine_distance(a,b):
    return(np.inner(a, b) / (norm(a) * norm(b)))

def euclidian_distance(a,b):
    return(norm(a-b)*-1)
    
def find_matches(pred, collection_features, files_and_features, nimages=8, distance='cosine'): 
#    img = kimage.load_img(imageurl, target_size=(224, 224))
#    x = kimage.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    img = preprocess_input(img)
#    pred = model.predict(img)
    pred = pred.flatten()
    
    nimages = len(collection_features)
    sims = np.zeros((nimages, 1))
    for i in range(0,nimages):
        if distance=='cosine':
            sims[i]= cosine_distance(pred.flatten(),collection_features[i].flatten())
        else:
            sims[i]= euclidian_distance(pred.flatten(),collection_features[i].flatten())
    similar_images=dict(zip(images,sims))
    topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=True)[0:nimages+1]
    return(topmatches)

        
def find_matches2(pred, collection_features, images, nimages=8, distance='cosine'): 
#    img = kimage.load_img(imageurl, target_size=(224, 224))
#    x = kimage.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    img = preprocess_input(img)
#    pred = model.predict(img)
    pred = pred.flatten()
    
    nimages = len(collection_features)
    sims = np.zeros((nimages, 1))
    for i in range(0,nimages):
        if distance=='cosine':
            sims[i]= cosine_distance(pred.flatten(),collection_features[i].flatten())
        else:
            sims[i]= euclidian_distance(pred.flatten(),collection_features[i].flatten())
    topmatches=pd.DataFrame(images,columns=['imgfiles'])
#    topmatches['simscore']=sims
    return(topmatches)
#%%
chosenimage="/home/adam/Pictures/sun1.jpg"
img = kimage.load_img(chosenimage, target_size=(224, 224))
img = kimage.img_to_array(img)
img = np.expand_dims(img, axis=0)    
pred=model.predict(img)
#%%
matches=find_matches(pred, collection_features, files_and_titles['imgfile'],nimages=5)

#%%
matches = pd.DataFrame(matches, columns=['imgfile', 'simscore'])
#%%
joinedpd = pd.merge(files_and_titles,matchespd)