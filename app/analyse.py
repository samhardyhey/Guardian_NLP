import string
import pandas as pd

import nltk
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans


# initialize constants, lematizer, punctuation and stopwords
lemmatizer = WordNetLemmatizer()
punct = set(string.punctuation)

# define stopwords
custom_stop_words = ['–', '\u2019', 'u', '\u201d', '\u201d.',
                     '\u201c', 'say', 'saying', 'sayings',
                     'says', 'us', 'un', '.\"', 'would',
                     'let', '.”', 'said', ',”'
                     ]
stopwords = set(sw.words('english') + custom_stop_words)


def lemmatize(token, tag):
    # collapse word inflections into single representation
    tag = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }.get(tag[0], wordnet.NOUN)

    return lemmatizer.lemmatize(token, tag)


def cab_tokenizer(document):
    # tokenize the corpus
    tokens = []

    # split the document into sentences
    for sent in sent_tokenize(document):
        # tokenize each sentence
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            # preprocess and remove unnecessary characters
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            # If punctuation, ignore token and continue
            if all(char in punct for char in token):
                continue

            # If stopword, ignore token and continue
            if token in stopwords:
                continue

            # Lemmatize the token and add back to the token
            lemma = lemmatize(token, tag)

            # Append lemmatized token to list
            tokens.append(lemma)
    return tokens


def retrieveTopTFIDF(df, main):
    # retrieve top-ranked terms via TFIDF vector
    df = df['bodyText']

    count_vec = CountVectorizer(tokenizer=cab_tokenizer,
                                ngram_range=(1, 2),
                                min_df=0.2, max_df=0.8)
    X_count = count_vec.fit_transform(df)

    totalTokens = len(count_vec.get_feature_names())

    # format terms
    terms = [{'term': t,
              'tf': int(X_count[:, count_vec.vocabulary_[t]].sum()),
              'df': int(X_count[:, count_vec.vocabulary_[t]].count_nonzero())}
             for t in count_vec.vocabulary_]

    topTenTerms = sorted(terms,
                         key=lambda k: (k['tf'], k['df']),
                         reverse=True)[:10]

    tokenSum = sum(term['tf'] for term in terms)

    # update main data object
    main.update({'totalTokens': totalTokens,
                 'tokenSum': tokenSum,
                 'topTenTerms': topTenTerms})

    return main


def createKMeans(df, main):
    # vectorize, fit and generate clusters
    tfidf_vec = TfidfVectorizer(tokenizer=cab_tokenizer,
                                ngram_range=(1, 2),
                                min_df=0.2, max_df=0.8)
    X = tfidf_vec.fit_transform(df['bodyText'])

    kmeans = KMeans(n_clusters=7, random_state=42).fit(X)

    # update main data object
    main['kMeanClusters'] = formatKMeans(kmeans.n_clusters,
                                         kmeans.cluster_centers_,
                                         tfidf_vec.get_feature_names())

    return main


def formatKMeans(n_clusters, cluster_centers, terms, num_word=5):
    # format results for prominent clusters
    ordered_centroids = cluster_centers.argsort()[:, ::-1]

    clusters = dict()

    for cluster in range(n_clusters):
        temp = []
        for term_idx in ordered_centroids[cluster, :5]:
            temp.append(terms[term_idx])
        clusters[cluster] = temp

    return clusters


def descriptive(df, main):
    # descriptive, macro information
    main['articleCount'] = len(df.index)

    main['totalChar'] = int(df['charCount'].sum())

    main['totalWord'] = int(df['wordcount'].sum())

    return main
