import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import time

# Scraping script from different newspapers:
# - El Pays -> get_news_elpays()
# - The Guardian -> get_news_guardian()
# - Daily Mail -> get_news_dailymail()
# - The Mirror -> get_news_themirror()
#
# The objective of this step is to obtain, for each newspaper:
# - A dataframe with the news content (input)
# - A dataframe with the article title, link and predicted category

def get_news_elpais(number_of_articles = 5):

    # url definition
    url = "https://english.elpais.com"
    
    # Request
    r1 = requests.get(url)
    r1.status_code

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('h2', class_='headline')

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    # If enough news to fetch
    if(len(coverpage_news)>=number_of_articles):

        for n in np.arange(0, number_of_articles):

            # Getting the link of the article
            link = coverpage_news[n].find('a')['href']
            list_links.append(link)

            # Getting the title
            title = coverpage_news[n].find('a').get_text()
            list_titles.append(title)

            # Reading the content (it is divided in paragraphs)
            article = requests.get(url + link)
            article_content = article.content
            soup_article = BeautifulSoup(article_content, 'html5lib')
            body = soup_article.find_all('div', class_='article_body')
            x = body[0].find_all('p')

            # Unifying the paragraphs
            list_paragraphs = [o.get_text() for o in x]            
            final_article = " ".join(list_paragraphs)

            news_contents.append(final_article)

    # df_features
    df_features = pd.DataFrame({'Content': news_contents})

    # df_show_info
    df_show_info = pd.DataFrame({'Article Title': list_titles, 'Article Link': list_links, 'Newspaper': 'El Pais English'})
    
    return (df_features, df_show_info)

def get_news_theguardian(number_of_articles = 5):
    
    # url definition
    url = "https://www.theguardian.com/uk"
    
    # Request
    r1 = requests.get(url)
    r1.status_code

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification (get <section> then <a> with the headlines)
    all_anchors = soup1.find('section', id='headlines').find_all('a', class_='js-headline-text')
    
    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, number_of_articles):

        # Getting the link of the article
        link = all_anchors[n]['href']
        list_links.append(link)

        # Getting the title
        title = all_anchors[n].get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('div', class_='content__article-body')
        if(len(body) > 0):
            x = body[0].find_all('p')

            # Unifying the paragraphs
            list_paragraphs = [o.get_text() for o in x]
            final_article = " ".join(list_paragraphs)

            news_contents.append(final_article)

    # df_features
    df_features = pd.DataFrame({'Content': news_contents})

    # df_show_info
    df_show_info = pd.DataFrame({'Article Title': list_titles, 'Article Link': list_links, 'Newspaper': 'The Guardian'})
    
    return (df_features, df_show_info)

def get_news_dailymail(number_of_articles = 5):
    
    # url definition
    url = "https://www.dailymail.co.uk"
    
    # Request
    r1 = requests.get(url)
    r1.status_code

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('h2', class_='linkro-darkred')

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, number_of_articles):

        # Getting the link of the article
        link = url + coverpage_news[n].find('a')['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('p', class_='mol-para-with-font')

        # Unifying the paragraphs
        list_paragraphs = [o.get_text() for o in body]
        final_article = " ".join(list_paragraphs)

         # Removing special characters
        final_article = re.sub("\\xa0", "", final_article)
        
        news_contents.append(final_article)
        

    # df_features
    df_features = pd.DataFrame({'Content': news_contents})

    # df_show_info
    df_show_info = pd.DataFrame({'Article Title': list_titles, 'Article Link': list_links, 'Newspaper': 'Daily Mail'})
    
    return (df_features, df_show_info)

def get_news_themirror(number_of_articles = 5):
    
    # url definition
    url = "http://www.mirror.co.uk/"
    
    # Request with simple single retry
    r1 = requests.get(url)
    i = 1
    while(r1.status_code != 200):
        time.sleep(i)
        r1 = requests.get(url)
        i = i + 1

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('a', class_='headline')

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, number_of_articles):

        # Getting the link of the article
        link = coverpage_news[n]['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('div', class_='article-body')
        x = soup_article.find_all('p')

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
        final_article = " ".join(list_paragraphs)

        news_contents.append(final_article)

    # df_features
    df_features = pd.DataFrame({'Content': news_contents})

    # df_show_info
    df_show_info = pd.DataFrame({'Article Title': list_titles, 'Article Link': list_links, 'Newspaper': 'El Pais English'})
    
    return (df_features, df_show_info)

if(__name__ == "__main__"):
    start = time.time()

    d_ElPaysFeatures, d_ElPaysInfo = get_news_elpais()
    d_GuardianFeatures, d_GuardianInfo = get_news_theguardian()
    d_DailyMailFeatures, d_DailyMailInfo = get_news_dailymail()
    d_TheMirrorFeatures, d_TheMirrorInfo = get_news_themirror()
    print(len(d_ElPaysFeatures), len(d_GuardianFeatures), len(d_DailyMailFeatures), len(d_TheMirrorFeatures))

    print("The time elapsed is %f seconds" %(time.time()-start))

