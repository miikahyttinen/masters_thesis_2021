from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urlparse
import random
import pandas as pd
import os


def getHTMLdocument(url, text=''):
    response = requests.get(url, timeout=2)
    return response.text


def build_link(href, domain):
    if domain in href:
        return href
    else:
        return 'https://' + domain + href


def extract_text(url):
    html_document = getHTMLdocument(url)
    # create soap object
    soup = BeautifulSoup(html_document)

    # kill all script and style elements
    for script in soup(["script", "style"]):
        # rip it out
        script.extract()

    text = soup.get_text()
    # print(text)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return ' '.join(chunk for chunk in chunks if chunk)


def scrape(domain, url_to_scrape):

    document = ''

    html_document = getHTMLdocument(url_to_scrape)

    # create soap object
    soup = BeautifulSoup(html_document, 'html.parser')

    # kill all script and style elements
    for script in soup(["script", "style"]):
        # rip it out
        script.extract()

    text = soup.get_text(' ')
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    document = document + text
    # visited = []
    # all_a = soup.find_all('a')
    # sample = 20 if len(all_a) > 20 else len(all_a)
    # upper_limit = len(all_a)
    # print('UPPER LIMIT', upper_limit)
    # it = random.sample(range(1, upper_limit), sample - 1)
    # print(it)
    # index = 0
    # for i in it:
    #    link_el = all_a[i]
    #    print(str(index) + '/' + str(len(it)))
    #    index = index + 1
    #    try:
    #        link = build_link(link_el.get('href'), domain)
    #        print(link)
    #    except:
    #        print('ERROR WITH LINK EL: ')
    #        continue
    #    if link not in visited:
    #        try:
    #            document = document + extract_text(link)
    #        except Exception as e:
    #            print(e)
    #            print('ERROR WITH URL: ' + link)
    #        continue
    return document


df = pd.read_csv('large_data_set/large_data_set.csv',
                 sep=';')
it = 0
for index, row in df.iterrows():
    print(index)
    try:
        URL = 'http://' + row['url']
        t = scrape(urlparse(URL).netloc.replace(
            'www.', ''), URL).replace('\n', ' ')
        if(len(t) < 100):
            print('ERROR: under 100 chars URL: ' + URL)
            continue
        data_row = str(row['industry_code']) + ';' + \
            str(row['business_id']) + ';' + \
            row['name'] + ';' + row['url'] + ';' + t
        f = open("large_data_set/data.txt", "a")
        f.write(data_row + '\n')
    except:
        print('ERROR: ' + URL)
        continue

f.close()
