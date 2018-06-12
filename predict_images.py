#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 21:42:33 2018

@author: adam
"""
#%%
def find_matches(imageurl,X,model,num_matches=10):
#    from keras.applications import VGG16
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing import image as kimage
    import numpy as np
    import scipy.sparse as sp
    def cosine_similarity(ratings):
        sim = ratings.dot(ratings.T)
        if not isinstance(sim, np.ndarray):
            sim = sim.toarray()
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return (sim / norms / norms.T)
    
    #model = VGG16(include_top=False, weights='imagenet')
    img = kimage.load_img(imageurl, target_size=(224, 224))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    n_dims = 25088
    shp = (1, n_dims)
    preds=sp.lil_matrix((1, n_dims))
    preds[0, :] = pred.reshape(shp)
    preds= preds.tocsr()
    X[0,:] = preds
    sim = cosine_similarity(X)
    return(sim[0])
    
#%%
import numpy as np   
    
X = np.load('/home/adam/artnetwork/savedfeatures.npy')
#X = X.all()
#%%
from keras.applications import VGG16
model = VGG16(include_top=False, weights='imagenet')
#%%
chosenimage = "/home/adam/Pictures/sun1.jpg"
simsun = find_matches(chosenimage,preds,model)
#%%
from IPython.display import display, Image
import glob, os
import operator
imagespath= "/home/adam/artnetwork/fineartamericaspider/output/full"
os.chdir(imagespath)
images=glob.glob("*.jpg")
#%%
images.insert(0,"/home/adam/Pictures/sun1.jpg")
similar_images=dict(zip(images,simsun[0]))

#%%
topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=True)[0:10]
for match in topmatches:
    display(Image(match[0]))


#%%
topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=False)[0:10]
for match in topmatches:
    display(Image(match[0]))


#%%

def display_matches(imageurl,preds,model,nimages=10):
    from IPython.display import display, Image
    import glob, os
    import operator
    imagespath= "/home/adam/artnetwork/fineartamericaspider/output/full"
    os.chdir(imagespath)    
    images=glob.glob("*.jpg")
    images.insert(0,imageurl)
    
    similarities = find_matches(chosenimage,preds,model)
    similar_images=dict(zip(images,similarities))
    topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=True)[0:nimages]
    for match in topmatches:
        display(Image(match[0]))

#%%
        
chosenimage = "/home/adam/Pictures/sun1.jpg"
display_matches(chosenimage,preds,model,5)

#%%
chosenimage = "/home/adam/Pictures/102374000.jpg"
display_matches(chosenimage,preds,model,20)


#%%
chosenimage = "/home/adam/Pictures/drawn-child-rainbow-2.jpg"
display_matches(chosenimage,preds,model,20)

#%%
chosenimage = "/home/adam/Pictures/sailboat.jpg"
display_matches(chosenimage,preds,model,20)

#%%
chosenimage = "/home/adam/Pictures/Anna_forrest.jpg"
display_matches(chosenimage,preds,model,20)

#%%
chosenimage = "/home/adam/Pictures/The-best-way-to-display-kids-artwork.png"
display_matches(chosenimage,preds,model,20)

