
# coding: utf-8

# ## Linking to images
# My web crawler downloaded images using the SHA1 hash as the filename. In this notebook, I am attempting to link the URL to the images so that I can then link to the site.

# In[1]:


import json
from hashlib import sha1
from IPython.display import Image
import pandas as pd
import numpy as np


# In[2]:


dataloc="/home/adam/artnetwork/fineartamericaspider/finearturls.json"
with open(dataloc) as train_file:
    dict_train = json.load(train_file)


# In[3]:


output=open(dataloc).read()
data = json.loads(output)


# In[4]:


d=data[0]


# In[5]:


filenames=list(d.values())


# In[35]:


art_title=filenames[0].split("/")[-1].split('.')[0]


# In[29]:


filenames[0]


# In[23]:


imgfile=sha1(filenames[0].encode('utf-8')).hexdigest()+".jpg"


# In[25]:


imgpath="/home/adam/artnetwork/fineartamericaspider/output/full/"


# In[28]:


Image(imgpath+imgfile)


# In[36]:


linkurl="https://fineartamerica.com/featured/" + art_title +".html?product=art-print"
linkurl


# In[5]:


def get_link_and_image(image):
    art_title=image.split("/")[-1].split('.')[0]
    return(dict({
    "art_title": art_title,
    "imgfile": sha1(image.encode('utf-8')).hexdigest()+".jpg",
    "linkurl": "https://fineartamerica.com/featured/" + art_title +".html?product=art-print",
    "image_url":image
    }))


# In[6]:


db=[]
for urls_list in data:
    for imageurls in list(urls_list.values()):
        for imageurl in imageurls:
            db.append(get_link_and_image(imageurl))


# In[7]:


len(db)


# In[88]:


imageur


# In[8]:


files_and_titles=pd.DataFrame(db).sort_values('art_title',).drop_duplicates()


# In[10]:


files_and_titles.to_csv('files_and_titles2.csv')


# In[9]:


collection_features = np.load('/home/adam/artnetwork/saved_collection_features.npy')


# In[117]:


type(collection_features)


# In[122]:


featureslist=[]
for features in collection_features:
    featureslist.append(features.flatten())


# In[ ]:


#featuresframe=pd.DataFrame([featureslist])


# In[12]:


files_and_titles=pd.read_csv('/home/adam/Downloads/files_and_titles.csv')


# In[23]:


testdf=files_and_titles.head(10)


# In[33]:


len(testdf)


# In[32]:


testdf.columns


