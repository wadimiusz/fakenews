import pandas as pd
import pickle
import requests
from tqdm.auto import tqdm
import matplotlib as plt
import seaborn as sns
from urllib.parse import quote
import time
import os
import requests
from bs4 import BeautifulSoup
import networkx as nx



URL = "https://www.googleapis.com/customsearch/v1"


def google(key, cx, query):
    PARAMS = {'key': key, 'cx': cx, 'q':query}
    r = requests.get(url = URL, params = PARAMS) 
    return r.json()

def dataprocessing(data):
    print("before data processing: " + str(data.shape))
    sns.distplot(data.title.apply(lambda x: len(x.split())), color="black")
    data["title_split"] = data.title.apply(lambda x: x.split())
    data["title_len"] = data.title.apply(lambda x: len(x.split()))
    data["title_url"] = data.title.apply(lambda x: quote(x, safe=""))
    data = data[(data.title_len >= 3) & (data.title_len <= 25)]
    print("after data processing: " + str(data.shape))
    data.reset_index(drop=True, inplace=True)
    return data

def load_urls(data):
    if os.path.exists('title2url.pkl'):
        with open('title2url.pkl', 'rb') as f:
            title2url = pickle.load(f)
    else:
        title2url = dict()
        
    key = os.environ.get("PY_GOOGLE_KEY")
    cx = os.environ.get("PY_GOOGLE_CX")
    titles_to_annotate = [title for title in data.title if title not in title2url]
    bar = tqdm(titles_to_annotate)
    fails = 0
    stopwords = ['kaggle', 'notebook', 'ipynb', 'github', 'fake', 'detection']
    for title in bar:
        query = '"{}"'.format(title)
        res = google(key, cx, query)
        if not 'items' in res: 
            fails += 1
            bar.set_postfix(fails=fails)
            continue
        itemlist = []
        for item in res['items']:
            if 'link' in item:
                link = item['link']
                if all(word not in link for word in stopwords):
                    itemlist.append(item['link'])
        title2url[title] = itemlist
        with open("title2url.pkl", "wb") as f:
            pickle.dump(title2url, f)
        time.sleep(1.1)
    print("so far so good")

def load_links(title2url):
    if os.path.exists('title2links.pkl'):
        with open('title2links.pkl', 'rb') as f:
            title2links = pickle.load(f)
    else:
        title2links = dict()
    
    pb = tqdm([title for title in title2url if title not in title2links])
    errors = 0
    for title in pb:
        res = requests.get(title2url[title])
        if res.status_code == 200:
            soup = BeautifulSoup(res.text)
            links = [x.get("href") for x in soup.find_all("a")]
            links = [x for x in links if x is not None and len(x) > 2]
            title2links[title] = links
        else:
            errors += 1
            pb.set_postfix(errors=errors)
        with open("title2url.pkl", "wb") as f:
            pickle.dump(title2url, f)
        time.sleep(1.1)


def load_network(title2url):
    g = nx.DiGraph(title2url)
    return g


def main():

    
    data = pd.read_csv("./data/fake_or_real_news.csv", index_col=0)
    #key = input('Your key:')
    #cx = input('Your cx:')
    data = dataprocessing(data)
    load_urls(data)
    with open("title2url.pkl", "rb") as f:
        title2url = pickle.load(f)
        network = load_network(title2url)
   #load_links(title2url)



if __name__ == '__main__':
    main()
