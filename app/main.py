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

    # write final analysis
    with open('output/results.json', 'w') as outfile:
        json.dump(main, outfile, ensure_ascii=False)

if __name__ == '__main__':
    main()
