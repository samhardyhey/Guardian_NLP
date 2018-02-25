# NLP analysis
import string
import time
import pandas as pd

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

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


def format_topics(model, feature_names, no_top_words, time_elapsed):
    # top words for topic within given decomposition model
    analysis = dict()
    for topic_idx, topic in enumerate(model.components_):
        topic_placeholder = "Topic {}".format(topic_idx)
        analysis[topic_placeholder] = [feature_names[i]
                                       for i in (-topic).argsort()[:no_top_words]]
    analysis['time_sec'] = time_elapsed
    return analysis


def nmf(df, main, topics):
    # NMF requires TFIDF vectorizer
    st = time.time()
    tfidf_vectorizer = TfidfVectorizer(tokenizer=cab_tokenizer, ngram_range=(1, 2),
                                       min_df=0.1, max_df=0.90)
    tfidf = tfidf_vectorizer.fit_transform(df['bodyText'])
    # fit, transform, retrieve features names
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # Non-Negative Matrix Factorization - fit model using tfidf vector
    nmf = NMF(n_components=topics, random_state=1, alpha=0.1,
              l1_ratio=0.5, init='nndsvd').fit(tfidf)
    et = time.time() - st

    main['nmf'] = format_topics(nmf, tfidf_feature_names, topics, et)
    return main


def lda(df, main, topics):
    # LDA requires Count Vectorizer
    st = time.time()
    tf_vectorizer = CountVectorizer(tokenizer=cab_tokenizer, ngram_range=(1, 2),
                                    min_df=0.1, max_df=0.90)
    tf = tf_vectorizer.fit_transform(df['bodyText'])
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Latent Dirilicht Analysis - fit the model using term frequency vector
    lda = LatentDirichletAllocation(n_components=topics, max_iter=5,
                                    learning_method='online', learning_offset=50,
                                    random_state=0).fit(tf)
    et = time.time() - st

    main['lda'] = format_topics(lda, tf_feature_names, topics, et)
    return main


def lsa(df, main, topics):
    # SVD becomes LSA when paired with tfidf vectorizer
    st = time.time()
    tfidf_vectorizer = TfidfVectorizer(tokenizer=cab_tokenizer, ngram_range=(1, 2),
                                       min_df=0.1, max_df=0.90)
    tfidf = tfidf_vectorizer.fit_transform(df['bodyText'])
    # fit, transform, retrieve features names
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    svd = TruncatedSVD(n_components=topics, algorithm='randomized',
                       n_iter=5, random_state=42).fit(tfidf)
    et = time.time() - st

    main['lsa'] = format_topics(svd, tfidf_feature_names, topics, et)
    return main


def descriptive(df, main):
    # overview of provided df, cast to regular integers for serialization
    main['articleCount'] = int(len(df.index))
    main['totalChar'] = int(df['charCount'].sum())
    main['totalWord'] = int(df['wordcount'].sum())

    return main


def all_analysis(df, label):
    # all desciptive stats, decomposition methods for a provided df
    main_analysis = dict()
    main_analysis['name'] = label
    main_analysis = descriptive(df, main_analysis)
    main_analysis = nmf(df, main_analysis, 10)
    main_analysis = lda(df, main_analysis, 10)
    main_analysis = lsa(df, main_analysis, 10)
    return main_analysis

# To Do
# df segmentation + analysis => author, time period, topic tagging
# programmatically: upload results, shutdown instance upon script completion
