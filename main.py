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
    PARAMS = {'key':'AIzaSyCQN6Zl9aIi1pNf6seBOrw1X50PxoYQV2Q', 'cx':'009131931706956872601:yjwrnbki4sp', 'q':query}
    r = requests.get(url = URL, params = PARAMS) 
    return r.json()

def main():
    if os.path.exists('title2url.pkl'):
        with open('title2url.pkl', 'rb') as f:
            title2url = pickle.load(f)
    else:
        title2url = dict()
    
    data = pd.read_csv("./data/fake_or_real_news.csv", index_col=0)
#     key = input('Your key:')
#     cx = input('Your cx:')
    key = "AIzaSyCQN6Zl9aIi1pNf6seBOrw1X50PxoYQV2Q"
    cx = "009131931706956872601:yjwrnbki4sp"
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
        
        time.sleep(0.2)
                

if __name__ == '__main__':
    main()