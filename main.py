# local modules
from retrieve import *
from clean import *
from analyse import *


def main():

    # retrieve and clean
    retrieve_articles()
    df = read_in()
    df = scrub(df)

    # analyse
    main = dict()
    main = descriptive(df, main)
    main = retrieveTopTFIDF(df, main)
    main = createKMeans(df, main)

    print(main)

if __name__ == '__main__':
    main()
