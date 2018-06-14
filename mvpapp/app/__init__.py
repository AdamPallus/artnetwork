#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:55:13 2018

@author: adam
"""


from flask import Flask



app = Flask(__name__)
app.config.from_object(__name__) 
app.config.update(dict(
UPLOAD_FOLDER = "/home/adam/artnetwork/mvpapp/app/static/img/tmp",
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
))


from app import routes