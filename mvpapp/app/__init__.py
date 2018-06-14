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
UPLOAD_FOLDER = "/home/adam/flaskapps/mvpapp/app/static/img/tmp"
))


from app import routes