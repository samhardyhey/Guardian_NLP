# NLP analysis
import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

# local modules
from retrieve import *
from clean import *

# initialize constants, lematizer, punctuation and stopwords
lemmatizer = WordNetLemmatizer()
punct = set(string.punctuation)

# define stopwords
stopwords = set(sw.words('english'))


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


def retrieveTopTDIDF(df):
    # index each term's Term Frequency and Inverse Document Frequency
    df = df['bodyText']  # text entries only

    print(df.head(5))
    print(df.shape)

    # use count vectorizer to find TF and DF of each term
    count_vec = CountVectorizer(tokenizer=cab_tokenizer,
                                ngram_range=(1, 2), min_df=0.2, max_df=0.8)
    X_count = count_vec.fit_transform(df)

    # return total number of tokenized words
    totalTokens = len(count_vec.get_feature_names())

    # cast numpy integers back to python integers
    terms = [{'term': t,
              'tf': X_count[:, count_vec.vocabulary_[t]].sum(),
              'df': X_count[:, count_vec.vocabulary_[t]].count_nonzero()}
             for t in count_vec.vocabulary_]

    topTenTerms = sorted(terms, key=lambda k: (
        k['tf'], k['df']), reverse=True)[:10]

    tokenSum = sum(term['tf'] for term in terms)

    # return top ten terms as well as sum of all tokenizations
    return topTenTerms, tokenSum, totalTokens


def truncateSVD(df):
    # apply singular value decomposition (matrix factorization), retrieve
    # prominent clusters
    df = df.filter(['bodyText'])  # text entries only

    # collapse into bag of words representation, limit extreme terms
    vector = TfidfVectorizer(tokenizer=cab_tokenizer,
                             ngram_range=(1, 2), min_df=0.1, max_df=0.9)
    matrix = vector.fit_transform(df)

    # generate truncated SVD usingp reivously generated matrix
    svd = TruncatedSVD(n_components=20, algorithm='randomized',
                       n_iter=5, random_state=42)
    svdTrans = svd.fit_transform(matrix)

    # sort by term weighting
    sorted_comp = svd.components_.argsort()[:, ::-1]
    terms = vector.get_feature_names()

    # fill with 10 SVD cluster entries
    clusterTerms = []

    for comp_num in range(10):
        clusterTerms.append([terms[i] for i in sorted_comp[comp_num, :5]])

    return clusterTerms


def main():
    #retrieve and clean
    retrieve_articles()
    df = read_in()
    df = scrub(df)

    # retrieve most frequent, weighted terms
    topTDIDFTerms, tokenSum, totalTokens = retrieveTopTDIDF(df)

    print(topTDIDFTerms, tokenSum, totalTokens)

    # retrieve clusters
    clusterTerms = truncateSVD(df)

if __name__ == '__main__':
    main()
