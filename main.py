import pandas as pd
import pickle
import requests
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import quote
import time
import os
import requests
from bs4 import BeautifulSoup
import networkx as nx
from urllib.parse import urlparse
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, auc
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM


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


def evalBinaryClassifier(model, x, y, labels=['Positives', 'Negatives']):
    '''
    Visualize the performance of  a Logistic Regression Binary Classifier.

    Displays a labelled Confusion Matrix, distributions of the predicted
    probabilities for both classes, the ROC curve, and F1 score of a fitted
    Binary Logistic Classifier. Author: gregcondit.com/articles/logr-charts

    Parameters
    ----------
    model : fitted scikit-learn model with predict_proba & predict methods
        and classes_ attribute. Typically LogisticRegression or
        LogisticRegressionCV

    x : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples
        in the data to be tested, and n_features is the number of features

    y : array-like, shape (n_samples,)
        Target vector relative to x.

    labels: list, optional
        list of text labels for the two classes, with the positive label first

    Displays
    ----------
    3 Subplots

    Returns
    ----------
    F1: float
    '''
    # model predicts probabilities of positive class
    p = model.predict_proba(x)
    if len(model.classes_) != 2:
        raise ValueError('A binary class problem is required')
    if model.classes_[1] == 1:
        pos_p = p[:, 1]
    elif model.classes_[0] == 1:
        pos_p = p[:, 0]

    # FIGURE
    plt.figure(figsize=[15, 4])

    # 1 -- Confusion matrix
    cm = confusion_matrix(y, model.predict(x))
    plt.subplot(131)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                     annot_kws={"size": 14}, fmt='g')
    cmlabels = ['True Negatives', 'False Positives',
                'False Negatives', 'True Positives']
    for i, t in enumerate(ax.texts):
        t.set_text(t.get_text() + "\n" + cmlabels[i])
    plt.title('Confusion Matrix', size=15)
    plt.xlabel('Predicted Values', size=13)
    plt.ylabel('True Values', size=13)

    # 2 -- Distributions of Predicted Probabilities of both classes
    df = pd.DataFrame({'probPos': pos_p, 'target': y})
    plt.subplot(132)
    plt.hist(df[df.target == 1].probPos, density=True, bins=25,
             alpha=.5, color='green', label=labels[0])
    plt.hist(df[df.target == 0].probPos, density=True, bins=25,
             alpha=.5, color='red', label=labels[1])
    plt.axvline(.5, color='blue', linestyle='--', label='Boundary')
    plt.xlim([0, 1])
    plt.title('Distributions of Predictions', size=15)
    plt.xlabel('Positive Probability (predicted)', size=13)
    plt.ylabel('Samples (normalized scale)', size=13)
    plt.legend(loc="upper right")

    # 3 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y, p[:, 1])
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(133)
    plt.plot(fp_rates, tp_rates, color='green',
             lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    # plot current decision point:
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(fp / (fp + tn), tp / (tp + fn), 'bo', markersize=8, label='Decision Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve', size=15)
    plt.legend(loc="lower right")
    plt.subplots_adjust(wspace=.3)
    plt.show()
    # Print and Return the F1 score
    tn, fp, fn, tp = [i for i in cm.ravel()]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1

def main():

    
    data = pd.read_csv("./data/fake_or_real_news.csv", index_col=0)
    #key = input('Your key:')
    #cx = input('Your cx:')
    data = dataprocessing(data)
    #load_urls(data)
    with open("title2url.pkl", "rb") as f:
        title2url = pickle.load(f)

    #load_links(title2url)
    title2urls_trunc = {key: [urlparse(x).netloc for x in value] for key, value in title2url.items()}
    G = nx.Graph()

    for sites in title2urls_trunc.values():
        G.add_edges_from(combinations(sites, 2))
    print("edges: " + str(G.number_of_edges()) + "\n" + "nodes: " + str(G.number_of_nodes()) + "\n" + "components: " + str(nx.number_connected_components(G)) + "\n" + "largest component :" + str(len(max(nx.connected_components(G), key=len))) + "\n" + "average clustering: " + str(nx.average_clustering(G)))
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    
    X, y = list(), list()
    nodes = list(G.nodes())
    
    for title, label in tqdm(zip(data.title, data.label)):
        if title not in title2url:
            continue
            
        adj_urls = title2urls_trunc[title]
        ids = [nodes.index(url) for url in adj_urls if url in nodes]
        if len(ids) == 0:
            continue
        vector = adjacency_matrix[ids].mean(0)
        X.append(vector)
        real = int(label == "REAL")
        y.append(real)
    
    X = np.vstack(X)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("Fitting a logistic regression")
    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(evalBinaryClassifier(clf, X_test, y_test))
    print("escape")
        

if __name__ == '__main__':
    main()
