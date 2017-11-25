import os
import pandas as pd
import json


def flatten(article):
    # collapse internal feature dictionary
    if isinstance(article, dict):
        temp = {}
        temp.update(article['fields'])
        article.pop('fields')
        temp.update(article)
        return temp


def read_in():
    # collapse all articles into single df
    files = os.listdir('./articles')
    articles = []

    # flatten and append each article
    for each in files:
        with open('./articles/' + each) as json_data:
            for article in json.load(json_data):
                articles.append(flatten(article))

    df = pd.DataFrame(articles)
    return df


def scrub(df):
    # include useful components only
    df = df.filter(['type', 'sectionId', 'webPublicationDate', 'webTitle',
                    'trailText', 'byline', 'wordcount', 'firstPublicationDate',
                    'bodyText', 'charCount'])

    # differentiate necessary columns
    df['webPublicationDate'] = pd.to_datetime(df['webPublicationDate'])
    df['firstPublicationDate'] = pd.to_datetime(df['firstPublicationDate'])

    df['wordcount'] = df['wordcount'].astype(int)
    df['charCount'] = df['charCount'].astype(int)

    return df
