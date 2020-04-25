import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import re
# print("dash", dash.__version__, " dash_core_components", dcc.__version__, " dash_html_components", html.__version__)

import NewsScraping as ns

path_models = "./0. Latest News Classifier/04. Model Training/Models/"

# SVM
path_svm = path_models + 'best_svc.pickle'
with open(path_svm, 'rb') as data:
    svc_model = pickle.load(data)

path_tfidf = "./0. Latest News Classifier/03. Feature Engineering/Pickles/tfidf.pickle"

with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)

category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4,
    'other':5
}

punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))

def create_features_from_df(df):
    
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    
    df['Content_Parsed_3'] = df['Content_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
        
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    
    wordnet_lemmatizer = WordNetLemmatizer()
    nrows = len(df)
    lemmatized_text_list = []
    for row in range(0, nrows):

        # Create an empty list containing lemmatized words
        lemmatized_list = []
        # Save the text and its words into an object
        text = df.loc[row]['Content_Parsed_4']
        text_words = text.split(" ")
        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        # Join the list
        lemmatized_text = " ".join(lemmatized_list)
        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    
    df['Content_Parsed_5'] = lemmatized_text_list
    
    df['Content_Parsed_6'] = df['Content_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
        
    df = df['Content_Parsed_6']
    df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    
    # TF-IDF
    features = tfidf.transform(df).toarray()
    
    return features


def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category

def predict_from_features(features):
        
    # Obtain the highest probability of the predictions for each article
    predictions_proba = svc_model.predict_proba(features).max(axis=1)    
    
    # Predict using the input model
    predictions_pre = svc_model.predict(features)

    # Replace prediction with 6 if associated cond. probability less than threshold
    predictions = []

    for prob, cat in zip(predictions_proba, predictions_pre):
        if prob > .65:
            predictions.append(cat)
        else:
            predictions.append(5)

    # Return result
    categories = [get_category_name(x) for x in predictions]
    
    return categories

def complete_df(df, categories):
    df['Prediction'] = categories
    return df

# Get the scraped dataframes
df_features, df_show_info = ns.get_news_elpais()

# Create features
features = create_features_from_df(df_features)

# Predict
predictions = predict_from_features(features)

# Put into dataset
df = complete_df(df_show_info, predictions)