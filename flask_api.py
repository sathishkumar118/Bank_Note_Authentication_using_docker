#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 17:46:38 2021

@author: coe
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__) 
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome"

@app.route('/predict')
def predict_note_auth():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis, entropy]])
    return f'The predicted value is {prediction}'

@app.route('/predict_file', methods = ["POST"])
def predict_note_auth_from_file():
    df_test = pd.read_csv(request.files.get("file"))
    predictions = classifier.predict(df_test)
    return f'The predicted values are {list(predictions)}'

@app.route('/predict_from_filepath')
def predict_note_auth_from_file_path():
    predictions = classifier.predict(pd.read_csv(request.args.get('filepath')).values)
    return f'The predicted values are {list(predictions)}'

if __name__=='__main__':
    app.run()