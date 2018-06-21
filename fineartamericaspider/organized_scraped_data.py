'''
function to convert scraped json file to extract file ames and links to images
'''
import json
from hashlib import sha1
import pandas as pd

import glob, os


# In[2]:


#dataloc="/home/adam/artnetwork/fineartamericaspider/finearturls.json"
dataloc="/home/adam/artnetwork/fineartamericaspider/celebimages.json"

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
#    "linkurl": "https://fineartamerica.com/featured/" + art_title +".html?product=art-print",
    "linkurl": "https://fineartamerica.com/featured/" + art_title +".html",
    "image_url":image
    }))


# In[6]:
#go into nested json structure to extract

db=[]
for urls_list in data:
    for imageurls in list(urls_list.values()):
        for imageurl in imageurls:
            db.append(get_link_and_image(imageurl))



# In[8]:
#remove extra cells
imagespath= "/home/adam/artnetwork/fineartamericaspider/output/full"
os.chdir(imagespath)
images=glob.glob("*.jpg")
#%%

files_and_titles=pd.DataFrame(db).sort_values('art_title',).drop_duplicates()
files_and_titles=files_and_titles[files_and_titles['imgfile'].isin(images)]
files_and_titles.sort_values(by='imgfile',inplace=True)
files_and_titles.reset_index(inplace=True)
# In[10]:


#files_and_titles.to_csv('files_and_titles_6-17.csv')
files_and_titles.to_csv('files_and_titles_celeb.csv')
