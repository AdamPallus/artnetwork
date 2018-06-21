
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


# In[8]:


files_and_titles=pd.DataFrame(db).sort_values('art_title',).drop_duplicates()


# In[10]:


files_and_titles.to_csv('files_and_titles2.csv')



