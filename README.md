## 1.0 Guardian Natural Language Processing ##
Natural Language Processing utilizing the Guardian API and NLTK. Articles are retrieved using the Guardian Open Platform API.  General, meta statistics pertaining to the corpus' word/char count are initially calculated. Corpus lemmatization and tokenization is performed using the NLTK library. 

An Sklearn CountVectorizer is used to extract the ten most prominent terms within the corpus, as filtered by their term frequency and document frequency. An Sklearn TFIDF Vectorizer is then used to perform Kmeans clustering upon the corpus, revealing natural term groupings.

## 2.0 Getting Started ##
Build and run the docker image in the usual fashion. Run the main python file. JSON files containing the articles for each day specified within the retrieval range is written within the articles directory. A JSON file containing the final analysis is written within the output directory.

## 3.0 Built With ##
Python 3.5.2, scipy 0.19.1, numpy 1.13.3, sklearn 0.19.0, nltk 3.2.5 and pandas 0.20.3.

## 4.0 Authors ##
Sam Hardy - Implementation.

## 4.1 Acknowledgments ##
Jim Hogan (QUT professor), for his docker instruction. Hendi Lie (QUT tutor), for his instruction regarding Sklearn NLP.

## 4.2 Resources ##
| Title/Author  | Link  |
| ------------- | ----- |
| "How to use the Guardian's API to download article data for content analysis" - Dan Nguyen     | https://gist.github.com/dannguyen/c9cb220093ee4c12b840 |