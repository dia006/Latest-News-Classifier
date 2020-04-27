import mechanize
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

def get_url(url, retries=0):
    """
    It get the url with a delay and retry if status code returned not 200.
    Default is no retry
    """
    # It seems mechanize is more slow but more reliable than requests so I changed
    # to this one as it act as a browser
    r1 = mechanize.urlopen(url)
    i = 1
    while(r1.code != 200 and i < retries):
        print("{} - status code {} - Wait {} seconds & Retry".format(url, r1.code, i))
        time.sleep(i)
        r1 = mechanize.urlopen(url)
        if(i < retries):
            i = i + 1
        else:
            print("{} - status code {} - Too many retries, abandon!".format(url, r1.code, i))
            break
    return(r1)

def get_news_elpais(number_of_articles = 5):

    # url definition
    url = "https://english.elpais.com"
    
    # Request
    r1 = get_url(url)

    # We'll save in coverpage the cover page content
    coverpage = r1.read()

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('h2', class_='headline')

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, min(len(coverpage_news), number_of_articles)):

        # Getting the link of the article and the web page
        link = url + coverpage_news[n].find('a')['href']
        article = get_url(link)
        # If article fetched
        if(article.code==200):
            list_links.append(link)

            # Getting the title
            title = coverpage_news[n].find('a').get_text()
            list_titles.append(title)

            # Reading the content (it is divided in paragraphs)
            article_content = article.read()
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
    r1 = get_url(url)

    # We'll save in coverpage the cover page content
    coverpage = r1.read()

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification (get <section> then <a> with the headlines)
    coverpage_news = soup1.find('section', id='headlines').find_all('a', class_='js-headline-text')
    
    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, min(len(coverpage_news), number_of_articles)):

        # Getting the link of the article
        link = coverpage_news[n]['href']
        article = get_url(link)
        # If article fetched
        if(article.code==200):
            list_links.append(link)

            # Getting the title
            title = coverpage_news[n].get_text()
            list_titles.append(title)

            # Reading the content (it is divided in paragraphs)
            article_content = article.read()
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
    r1 = get_url(url)

    # We'll save in coverpage the cover page content
    coverpage = r1.read()

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('h2', class_='linkro-darkred')

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, min(len(coverpage_news), number_of_articles)):

        # Getting the link of the article
        link = url + coverpage_news[n].find('a')['href']
        article = get_url(link)
        # If article fetched
        if(article.code==200):
            list_links.append(link)

            # Getting the title
            title = coverpage_news[n].find('a').get_text()
            list_titles.append(title)

            # Reading the content (it is divided in paragraphs)
            article_content = article.read()
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
    url = "https://www.mirror.co.uk/"
    
    # Request with simple single retry
    r1 = get_url(url)

    # We'll save in coverpage the cover page content
    coverpage = r1.read()

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('a', class_='headline')

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, min(len(coverpage_news), number_of_articles)):

        # Getting the link of the article
        link = coverpage_news[n]['href']
        article = get_url(link)
        # If article fetched
        if(article.code==200):
            list_links.append(link)

            # Getting the title
            title = coverpage_news[n].get_text()
            list_titles.append(title)

            # Reading the content (it is divided in paragraphs)
            article = get_url(link)
            article_content = article.read()
            soup_article = BeautifulSoup(article_content, 'html5lib')
            body = soup_article.find_all('div', class_='article-body')
            x = soup_article.find_all('p')

            # Unifying the paragraphs
            list_paragraphs = [o.get_text() for o in x]
            final_article = " ".join(list_paragraphs)

            news_contents.append(final_article)

    # df_features
    df_features = pd.DataFrame({'Content': news_contents})

    # df_show_info
    df_show_info = pd.DataFrame({'Article Title': list_titles, 'Article Link': list_links, 'Newspaper': 'The Mirror'})
    
    return (df_features, df_show_info)

# Sky News
def get_news_skynews(number_of_articles = 5):
    
    # url definition
    url = "https://news.sky.com/us"

    # Request
    r1 = get_url(url)

    # We'll save in coverpage the cover page content
    coverpage = r1.read()

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('h3', class_="sdc-site-tile__headline")

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, min(len(coverpage_news), number_of_articles)):

        # Getting the link of the article
        link = "https://news.sky.com" + coverpage_news[n].find('a', class_='sdc-site-tile__headline-link')['href']
        article = get_url(link)
        # If article fetched
        if(article.code==200):
            # If video url discard it, this will lead to a lower number of news
            if("/video/" in link):
                continue

            list_links.append(link)

            # Getting the title
            title = coverpage_news[n].find('a').find('span').get_text()
            list_titles.append(title)

            # Reading the content (it is divided in paragraphs)
            article_content = article.read()
            soup_article = BeautifulSoup(article_content, 'html5lib')
            body = soup_article.find_all('div', class_='sdc-article-body')
            x = body[0].find_all('p')

            # Unifying the paragraphs
            list_paragraphs = [o.get_text() for o in x]
            final_article = " ".join(list_paragraphs)

            news_contents.append(final_article)

    # df_features
    df_features = pd.DataFrame({'Content': news_contents})

    # df_show_info
    df_show_info = pd.DataFrame({'Article Title': list_titles, 'Article Link': list_links, 'Newspaper': 'Sky News'})
    
    return (df_features, df_show_info)

if(__name__ == "__main__"):
    start = time.time()

    max_news = 10

    d_ElPaysFeatures, d_ElPaysInfo = get_news_elpais(max_news)
    print("El Pays", len(d_ElPaysFeatures), "on", max_news)

    d_GuardianFeatures, d_GuardianInfo = get_news_theguardian(max_news)
    print("The Guardian", len(d_GuardianFeatures), "on", max_news)

    d_DailyMailFeatures, d_DailyMailInfo = get_news_dailymail(max_news)
    print("Daily Mail", len(d_DailyMailFeatures), "on", max_news)

    d_TheMirrorFeatures, d_TheMirrorInfo = get_news_themirror(max_news)
    print("The Mirror", len(d_TheMirrorFeatures), "on", max_news)

    d_SkyNewsFeatures, d_SkyNewsInfo = get_news_skynews(max_news)
    print("Sky News", len(d_SkyNewsFeatures), "on", max_news)

    print("The time elapsed is %f seconds" %(time.time()-start))

