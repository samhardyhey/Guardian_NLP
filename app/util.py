# article compilation and sanitization, util functions
import os
import pandas as pd
import json


def flatten(article):
    # collapse internal article dictionary
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
        try:  # ensure parsing is encoding-resistant..
            with open('./articles/' + each) as json_data:
                for article in json.load(json_data):
                    articles.append(flatten(article))
        except:
            pass

    df = pd.DataFrame(articles)
    return df


def scrub(df):
    # include useful components only
    df = df.filter(['type', 'sectionId', 'webPublicationDate', 'webTitle',
                    'trailText', 'byline', 'wordcount', 'firstPublicationDate',
                    'bodyText', 'charCount'])

    # cast date/time types
    df['webPublicationDate'] = pd.to_datetime(df['webPublicationDate'])
    df['firstPublicationDate'] = pd.to_datetime(df['firstPublicationDate'])

    # generate meta information for each article
    df['wordcount'] = df['wordcount'].astype(int)
    df['charCount'] = df['charCount'].astype(int)

    df.dropna()  # all nan and 0 field entries
    df = df.loc[(df != 0).any(axis=1)]

    return df


def export_frame(file_name, df):
    # single frame export
    df.to_csv(file_name, index=False)


def write_json(file_name, results):
    # export results dict
    with open(file_name, 'w') as f:
        json.dump(results, f)


def retrieve_key():
    # guardian API key
    with open('./keys/guardian_key.txt') as f:
        data = f.readlines()
    return data[0]
