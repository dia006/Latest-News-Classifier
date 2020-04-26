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

import NewsScraping as ns
import Utilities as uti

# Get the scraped dataframes
df_features, df_show_info = ns.get_news_elpais()
features = uti.create_features_from_df(df_features)
predictions = uti.predict_from_features(features)
df = uti.complete_df(df_show_info, predictions)
print(df)

df_features, df_show_info = ns.get_news_theguardian()
features = uti.create_features_from_df(df_features)
predictions = uti.predict_from_features(features)
df = uti.complete_df(df_show_info, predictions)
print(df)

df_features, df_show_info = ns.get_news_dailymail()
features = uti.create_features_from_df(df_features)
predictions = uti.predict_from_features(features)
df = uti.complete_df(df_show_info, predictions)
print(df)

df_features, df_show_info = ns.get_news_themirror()
features = uti.create_features_from_df(df_features)
predictions = uti.predict_from_features(features)
df = uti.complete_df(df_show_info, predictions)
print(df)