# article retrieval
import json
import requests
import os
from os import mkdir
from os.path import join, exists
from datetime import timedelta
from util import retrieve_key

# define dev key and query object
MY_API_KEY = retrieve_key()
API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {
    'from-date': "",
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 50,
    'production-office': 'AUS',
    'lang': 'en',
    'page': 1,
    'api-key': MY_API_KEY
}


def retrieve_articles(start_date, end_date):
    dayrange = range((end_date - start_date).days + 1)

    # store articles
    ARTICLES_DIR = 'articles'

    for daycount in dayrange:
        dt = start_date + timedelta(days=daycount)
        datestr = dt.strftime('%Y-%m-%d')
        fname = join(ARTICLES_DIR, datestr + '.json')

        if not exists(fname):
            # then let's download it
            print("Downloading", datestr)
            all_results = []
            my_params['from-date'] = datestr
            my_params['to-date'] = datestr
            current_page = 1
            total_pages = 1

            while current_page <= total_pages:
                print("...page", current_page)
                my_params['page'] = current_page
                resp = requests.get(API_ENDPOINT, my_params)
                data = resp.json()
                all_results.extend(data['response']['results'])
                # if there is more than one page
                current_page += 1
                total_pages = data['response']['pages']

            with open(fname, 'w') as f:
                print("Writing to", fname)
                # re-serialize it for pretty indentation
                f.write(json.dumps(all_results, indent=2))
