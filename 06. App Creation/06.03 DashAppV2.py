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
import dash_renderer
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import re

import NewsScraping as ns
import Utilities as uti

# Stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Colors
colors = {
    'background': '#ECECEC',  
    'text': '#696969',
    'titles': '#599ACF',
    'blocks': '#F7F7F7',
    'graph_background': '#F7F7F7',
    'banner': '#C3DCF2'

}

# Markdown text
markdown_text1 = '''

This application gathers the latest news from the newspapers **El Pais**, **The Guardian** and **Sky News**, predicts their category between **Politics**, **Business**, **Entertainment**, **Sport**, **Tech** and **Other** and then shows a summary.

The scraped news are converted into a numeric feature vector with *TF-IDF vectorization*. Then, a *Support Vector Classifier* is applied to predict each category.

This app is meant for didactic purposes.

Please enter which newspapers would you like to scrape news off and press the **Scrape** button.

'''

markdown_text2 = '''

 Created by Miguel Fernández Zafra. Visit my webpage at [mfz.es](https://www.mfz.es/) and the [github repo](https://github.com/miguelfzafra/Latest-News-Classifier).

 *Disclaimer: this app is not under periodic maintenance. A live web-scraping process is carried out every time you run the app, so there may be some crashes due to the failing status of some requests.*

'''



app.layout = html.Div(style={'backgroundColor':colors['background']}, children=[
    
    # Space before title
    html.H1(children=' ',
            style={'padding': '10px'}
           ),
    
    # Title
    html.Div(
        [
            html.H3(children='News Classification App',
                    style={"margin-bottom": "0px"}
                   ),
            html.H6(children='A Machine Learning based app')
        ],
        style={
            'textAlign': 'center',
            'color': colors['text'],
            #'padding': '0px',
            'backgroundColor': colors['background']
              },
        className='banner',
            ),
    

    # Space after title
    html.H1(children=' ',
            style={'padding': '1px'}),


    # Text boxes
    html.Div(
        [
            html.Div(
                [
                    html.H6(children='What does this app do?',
                            style={'color':colors['titles']}),
                    
                    html.Div(
                        [dcc.Markdown(children=markdown_text1),],
                        style={'font-size': '12px',
                               'color': colors['text']}),
                                        
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=[
                                    {'label': 'El Pais English', 'value': 'EPE'},
                                    {'label': 'The Guardian', 'value': 'THG'},
                                    {'label': 'The Mirror', 'value': 'TMI'},
                                    {'label': 'Daily Mail', 'value': 'DMI'},
                                    {'label': 'Sky News', 'value': 'SKN'}
                                ],
                                value=['EPE', 'THG', 'TMI', 'DMI', 'SKN'],
                                multi=True,
                                id='checklist'),
                        ],
                        style={'font-size': '12px',
                               'margin-top': '25px'}),
                    
                    html.Div([
                        html.Button('Scrape', 
                                    id='submit', 
                                    type='submit', 
                                    style={'color': colors['blocks'],
                                           'background-color': colors['titles'],
                                           'border': 'None'})],
                        style={'textAlign': 'center',
                               'padding': '20px',
                               "margin-bottom": "10px",
                               'color': colors['titles']}),
            
                    dcc.Loading(id="loading-1", children=[html.Div(id="loading-output-1")], type="circle"),
                    
                    html.Hr(),
                    html.H6(children='Headlines',
                            style={'color': colors['titles']}),

                    # Headlines
                    html.A(id="textarea1a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea1b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea2a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea2b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea3a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea3b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea4a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea4b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea5a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea5b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea6a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea6b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea7a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea7b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea8a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea8b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea9a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea9b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea10a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea10b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea11a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea11b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea12a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea12b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea13a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea13b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea14a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea14b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea15a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea15b", style={'color': colors['text'], 'font-size': '11px'})
                                                            
                ],
                     style={'backgroundColor': colors['blocks'],
                            'padding': '20px',
                            'border-radius': '5px',
                            'box-shadow': '1px 1px 1px #9D9D9D'},
                     className='one-half column'),
            
            html.Div(
                [
                    html.H6("Graphic summary",
                            style={'color': colors['titles']}),

                    html.Div([
                         dcc.Graph(id='graph1', style={'height': '300px'})
                         ],
                         style={'backgroundColor': colors['blocks'],
                                'padding': '20px'}
                    ),
                    
                    html.Div([
                         dcc.Graph(id='graph2', style={'height': '300px'})
                         ],
                         style={'backgroundColor': colors['blocks'],
                                'padding': '20px'}
                    )
                ],
                     style={'backgroundColor': colors['blocks'],
                            'padding': '20px',
                            'border-radius': '5px',
                            'box-shadow': '1px 1px 1px #9D9D9D'},
                     className='one-half column')

        ],
        className="row flex-display",
        style={'padding': '20px',
               'margin-bottom': '0px'}
    ),
    
        
    # Space
    html.H1(id='space2', children=' '),
        
    
    # Final paragraph
    html.Div(
            [dcc.Markdown(children=markdown_text2),],
            style={'font-size': '12px',
                   'color': colors['text']}),

    
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'})
])


@app.callback(
    [
    Output('intermediate-value', 'children'),
    Output('loading-1', 'children')
    ],
    [Input('submit', 'n_clicks')],
    [State('checklist', 'value')])
def scrape_and_predict(n_clicks, values):
            
    df_features = pd.DataFrame()
    df_show_info = pd.DataFrame()
    
    # if(max_news is None or max_news < 5):
    #     max_news = 5
    max_news = 5
    
    if 'EPE' in values:
        # Get the scraped dataframes
        d1, d2 = ns.get_news_elpais(max_news)
        df_features = df_features.append(d1)
        df_show_info = df_show_info.append(d2)
    
    if 'THG' in values:
        d1, d2 = ns.get_news_theguardian(max_news)
        df_features = df_features.append(d1)
        df_show_info = df_show_info.append(d2)
        
    if 'TMI' in values:
        d1, d2 = ns.get_news_themirror(max_news)
        df_features = df_features.append(d1)
        df_show_info = df_show_info.append(d2)

    if 'DMI' in values:
        d1, d2 = ns.get_news_dailymail(max_news)
        df_features = df_features.append(d1)
        df_show_info = df_show_info.append(d2)

    if 'SKN' in values:
        d1, d2 = ns.get_news_skynews(max_news)
        df_features = df_features.append(d1)
        df_show_info = df_show_info.append(d2)

    df_features = df_features.reset_index().drop('index', axis=1)
    
    # Create features
    features = uti.create_features_from_df(df_features)
    # Predict
    predictions = uti.predict_from_features(features)
    # Put into dataset
    df = uti.complete_df(df_show_info, predictions)
    # df.to_csv('Tableau Teaser/df_tableau.csv', sep='^')  # export to csv to work out an example in Tableau
    
    return df.to_json(date_format='iso', orient='split'), ' '

@app.callback(
    Output('graph1', 'figure'),
    [Input('intermediate-value', 'children')])
def update_barchart(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    
    # Create a summary df
    df_sum = df.groupby(['Newspaper', 'Prediction']).count()['Article Title']

    # Create x and y arrays for the bar plot for every newspaper
    if 'El Pais English' in df_sum.index:
    
        df_sum_epe = df_sum['El Pais English']
        x_epe = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_epe = [[df_sum_epe['politics'] if 'politics' in df_sum_epe.index else 0][0],
                [df_sum_epe['business'] if 'business' in df_sum_epe.index else 0][0],
                [df_sum_epe['entertainment'] if 'entertainment' in df_sum_epe.index else 0][0],
                [df_sum_epe['sport'] if 'sport' in df_sum_epe.index else 0][0],
                [df_sum_epe['tech'] if 'tech' in df_sum_epe.index else 0][0],
                [df_sum_epe['other'] if 'other' in df_sum_epe.index else 0][0]]   
    else:
        x_epe = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_epe = [0,0,0,0,0,0]
    
    if 'The Guardian' in df_sum.index:
        
        df_sum_thg = df_sum['The Guardian']
        x_thg = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_thg = [[df_sum_thg['politics'] if 'politics' in df_sum_thg.index else 0][0],
                [df_sum_thg['business'] if 'business' in df_sum_thg.index else 0][0],
                [df_sum_thg['entertainment'] if 'entertainment' in df_sum_thg.index else 0][0],
                [df_sum_thg['sport'] if 'sport' in df_sum_thg.index else 0][0],
                [df_sum_thg['tech'] if 'tech' in df_sum_thg.index else 0][0],
                [df_sum_thg['other'] if 'other' in df_sum_thg.index else 0][0]]   
    else:
        x_thg = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_thg = [0,0,0,0,0,0]

    if 'The Mirror' in df_sum.index:
    
        df_sum_tmi = df_sum['The Mirror']
        x_tmi = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_tmi = [[df_sum_tmi['politics'] if 'politics' in df_sum_tmi.index else 0][0],
                [df_sum_tmi['business'] if 'business' in df_sum_tmi.index else 0][0],
                [df_sum_tmi['entertainment'] if 'entertainment' in df_sum_tmi.index else 0][0],
                [df_sum_tmi['sport'] if 'sport' in df_sum_tmi.index else 0][0],
                [df_sum_tmi['tech'] if 'tech' in df_sum_tmi.index else 0][0],
                [df_sum_tmi['other'] if 'other' in df_sum_tmi.index else 0][0]]   

    else:
        x_tmi = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_tmi = [0,0,0,0,0,0]

    if 'Daily Mail' in df_sum.index:
    
        df_sum_dm = df_sum['Daily Mail']
        x_dm = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_dm = [[df_sum_dm['politics'] if 'politics' in df_sum_dm.index else 0][0],
                [df_sum_dm['business'] if 'business' in df_sum_dm.index else 0][0],
                [df_sum_dm['entertainment'] if 'entertainment' in df_sum_dm.index else 0][0],
                [df_sum_dm['sport'] if 'sport' in df_sum_dm.index else 0][0],
                [df_sum_dm['tech'] if 'tech' in df_sum_dm.index else 0][0],
                [df_sum_dm['other'] if 'other' in df_sum_dm.index else 0][0]]   

    else:
        x_dm = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_dm = [0,0,0,0,0,0]

    if 'Sky News' in df_sum.index:
    
        df_sum_skn = df_sum['Sky News']
        x_skn = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_skn = [[df_sum_skn['politics'] if 'politics' in df_sum_skn.index else 0][0],
                [df_sum_skn['business'] if 'business' in df_sum_skn.index else 0][0],
                [df_sum_skn['entertainment'] if 'entertainment' in df_sum_skn.index else 0][0],
                [df_sum_skn['sport'] if 'sport' in df_sum_skn.index else 0][0],
                [df_sum_skn['tech'] if 'tech' in df_sum_skn.index else 0][0],
                [df_sum_skn['other'] if 'other' in df_sum_skn.index else 0][0]]   

    else:
        x_skn = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_skn = [0,0,0,0,0,0]

    # Create plotly figure
    figure = {
        'data': [
            {'x': x_epe, 'y':y_epe, 'type': 'bar', 'name': 'El Pais', 'marker': {'color': 'rgb(1, 108, 162)'}},
            {'x': x_thg, 'y':y_thg, 'type': 'bar', 'name': 'The Guardian', 'marker': {'color': 'rgb(5, 41, 98)'}},
            {'x': x_tmi, 'y':y_tmi, 'type': 'bar', 'name': 'The Mirror', 'marker': {'color': 'rgb(233, 14, 14)'}},
            {'x': x_dm, 'y':y_dm, 'type': 'bar', 'name': 'Daily Mail', 'marker': {'color': 'rgb(0, 76, 179)'}},
            {'x': x_skn, 'y':y_skn, 'type': 'bar', 'name': 'Sky News', 'marker': {'color': 'rgb(171, 0, 0)'}}
        ],
        'layout': {
            'title': 'Number of news articles by newspaper',
            'plot_bgcolor': colors['graph_background'],
            'paper_bgcolor': colors['graph_background'],
            'font': {
                    'color': colors['text'],
                    'size': '10'
            },
            'barmode': 'stack'
            
        }   
    }

    return figure

@app.callback(
    Output('graph2', 'figure'),
    [Input('intermediate-value', 'children')])
def update_piechart(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    
    # Create a summary df
    df_sum = df['Prediction'].value_counts()

    # Create x and y arrays for the bar plot
    x = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
    y = [[df_sum['politics'] if 'politics' in df_sum.index else 0][0],
         [df_sum['business'] if 'business' in df_sum.index else 0][0],
         [df_sum['entertainment'] if 'entertainment' in df_sum.index else 0][0],
         [df_sum['sport'] if 'sport' in df_sum.index else 0][0],
         [df_sum['tech'] if 'tech' in df_sum.index else 0][0],
         [df_sum['other'] if 'other' in df_sum.index else 0][0]]
    
    # Create plotly figure
    figure = {
        'data': [
            {'values': y,
             'labels': x, 
             'type': 'pie',
             'hole': .4,
             'name': '% of news articles',
             'marker': {'colors': ['rgb(248, 125, 37)',  # Politics
                                   'rgb(139, 51, 239)',  # Business
                                   'rgb(242, 211, 41)',  # Entertainment
                                   'rgb(44, 164, 84)',   # Sport
                                   'rgb(47, 69, 251)',   # Tech
                                   'rgb(246, 10, 34)']}, # Other

            }
        ],
        
        'layout': {
            'title': 'News articles by newspaper',
            'plot_bgcolor': colors['graph_background'],
            'paper_bgcolor': colors['graph_background'],
            'font': {
                    'color': colors['text'],
                    'size': '10'
            }
        }
        
    }
    
    return figure
    
    
@app.callback(
    [
    Output('textarea1a', 'href'),
    Output('textarea1a', 'children'),
    Output('textarea1b', 'children'),
    Output('textarea2a', 'href'),
    Output('textarea2a', 'children'),
    Output('textarea2b', 'children'),
    Output('textarea3a', 'href'),
    Output('textarea3a', 'children'),
    Output('textarea3b', 'children'),
    Output('textarea4a', 'href'),
    Output('textarea4a', 'children'),
    Output('textarea4b', 'children'),
    Output('textarea5a', 'href'),
    Output('textarea5a', 'children'),
    Output('textarea5b', 'children'),
    Output('textarea6a', 'href'),
    Output('textarea6a', 'children'),
    Output('textarea6b', 'children'),
    Output('textarea7a', 'href'),
    Output('textarea7a', 'children'),
    Output('textarea7b', 'children'),
    Output('textarea8a', 'href'),
    Output('textarea8a', 'children'),
    Output('textarea8b', 'children'),
    Output('textarea9a', 'href'),
    Output('textarea9a', 'children'),
    Output('textarea9b', 'children'),
    Output('textarea10a', 'href'),
    Output('textarea10a', 'children'),
    Output('textarea10b', 'children'),
    Output('textarea11a', 'href'),
    Output('textarea11a', 'children'),
    Output('textarea11b', 'children'),
    Output('textarea12a', 'href'),
    Output('textarea12a', 'children'),
    Output('textarea12b', 'children'),
    Output('textarea13a', 'href'),
    Output('textarea13a', 'children'),
    Output('textarea13b', 'children'),
    Output('textarea14a', 'href'),
    Output('textarea14a', 'children'),
    Output('textarea14b', 'children'),
    Output('textarea15a', 'href'),
    Output('textarea15a', 'children'),
    Output('textarea15b', 'children')
    ],
    [Input('intermediate-value', 'children')])
def update_textarea1(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    
    texts = []
    links = []
    preds_newsp = []
    
    for article in range(len(df)):
        texts.append(df.iloc[article]['Article Title'])
        links.append(df.iloc[article]['Article Link'])
        preds_newsp.append((df.iloc[article]['Prediction'].capitalize()) + ', ' + (df.iloc[article]['Newspaper']))

    while (len(texts) < 16):
        texts.append(None)
        links.append(None)
        preds_newsp.append(None)
    
    return \
        links[0], texts[0], preds_newsp[0],\
        links[1], texts[1], preds_newsp[1],\
        links[2], texts[2], preds_newsp[2],\
        links[3], texts[3], preds_newsp[3],\
        links[4], texts[4], preds_newsp[4],\
        links[5], texts[5], preds_newsp[5],\
        links[6], texts[6], preds_newsp[6],\
        links[7], texts[7], preds_newsp[7],\
        links[8], texts[8], preds_newsp[8],\
        links[9], texts[9], preds_newsp[9],\
        links[10], texts[10], preds_newsp[10],\
        links[11], texts[11], preds_newsp[11],\
        links[12], texts[12], preds_newsp[12],\
        links[13], texts[13], preds_newsp[13],\
        links[14], texts[14], preds_newsp[14]
           
    
    
# Loading CSS
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})

if(__name__=="__main__"):
    app.run_server(debug=False)