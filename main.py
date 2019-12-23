import pandas as pd
import pickle
import requests
from tqdm.auto import tqdm
import matplotlib as plt
import seaborn as sns
from urllib.parse import quote
import time
import os

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

def main():
    if os.path.exists('title2url.pkl'):
        with open('title2url.pkl', 'rb') as f:
            title2url = pickle.load(f)
    else:
        title2url = dict()
    
    data = pd.read_csv("./data/fake_or_real_news.csv", index_col=0)
    #key = input('Your key:')
    #cx = input('Your cx:')
    key = os.environ.get("PY_GOOGLE_KEY")
    cx = os.environ.get("PY_GOOGLE_CX")
    data = dataprocessing(data)
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
        for item in res['items']:
            if 'link' in item:
                link = item['link']
                if all(word not in link for word in stopwords):
                    title2url[title] = item['link']
                    break
        with open("title2url.pkl", "wb") as f:
            pickle.dump(title2url, f)
        time.sleep(1.1)
    print("so far so good")


if __name__ == '__main__':
    main()
