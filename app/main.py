from retrieve import retrieve_articles
from util import read_in, scrub, write_json, export_frame
from analyse import all_analysis
from datetime import date


def main():
    # 1.0 retrieve
    retrieve_articles(date(2017, 3, 1), date(2017, 3, 2))

    # 2.0 clean
    df = read_in()
    df = scrub(df)
    export_frame('./output/corpus.csv', df)

    # 3.0 analyse
    analysis = all_analysis(df, 'all_articles')

    # 4.0 write results
    write_json('./output/analysis.json', analysis)

if __name__ == '__main__':
    main()
