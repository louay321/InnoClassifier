"""In this module i used unsupervised ML to have a deep insight about the emails related topics
and to extract features which will be used for training the classifier.
"""

from modules.chunks import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

"""Init phase"""
# initialize language for spaCy
sp = spacy.load("en_core_web_sm")

""" get the emails body,sender and receiver from chunks.py 
after modifying them and getting rid of unwanted data
"""
emails_body = emails_df_body()
emails_to = emails_df_to()
emails_from_ = emails_df_from_()

# Created a customized list of stopwords :
stopwords_list = set(stopwords.words("english") + ['hou', 'ect', 'com', 'recipient'])


"""Here we use all the tools above to filter the emails body by 
tokenizing words, removing the Stop words from them and lemmatizing the rest.
"""


def email_body_filtering(email):
    email_tokenized = word_tokenize(email)
    email_nostopwords = [word for word in email_tokenized if not word in stopwords_list]
    email_string = ' '.join(email_nostopwords)
    email_filtered = ' '.join([token.lemma_ for token in sp(email_string)])

    return email_filtered


"""This method will call the above method to filter each body of 
the email and put it together with its sender and receiver data.
"""


def newParse(emails_body, emails_to, emails_from_):
    emails_body_new = (email_body_filtering(email) for email in emails_body)
    emails_new = {
        'body': (body for body in emails_body_new),
        'to': (to for to in emails_to),
        'from_': (from_ for from_ in emails_from_)
    }
    return emails_new


# create a new dataframe for modified bodies
def df_new():
    emails_new_df = pd.DataFrame(newParse(emails_body))
    return emails_new_df


"""Analyze the text using TF-IDF to check importance of words in documents"""
# initialize the vector
vect = TfidfVectorizer(max_df=0.50, min_df=2)
df_body = df_new().body
X = vect.fit_transform(df_body)


"""this method will make it possible to check the highest
and most frequent keywords 'features'
"""

def top_frq_features(row, features, view_num=20):
    topn_ids = np.argsort(row)[::-1][:view_num]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df


# to find feats in a doc :
def top_frq_feats_doc(X, features, row_id, view_num=25):
    row = np.squeeze(X[row_id].toarray())
    return top_frq_features(row, features, view_num)


features = vect.get_feature_names()
# print(top_frq_feats_doc(X, features, 1, 25))


# For each email this function extracts highest frq feats
def top_frq_means(X, features, grp_ids=None, min_tfidf=0.1, view_num=25):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_frq_features(tfidf_means, features, view_num)


# print(top_frq_means(X, features, view_num=10))

"""For the clustering here i used KMeans cluster
 which depends on the number of clusters given
"""


def labels_create():
    num_clusters = 6
    clf = KMeans(n_clusters=num_clusters, max_iter=100, init='k-means++', n_init=1)
    labels = clf.fit_predict(X)
    return labels


# view the clustered data
def view_data():
    X_dense = X.todense()
    coordinates = PCA(n_components=2).fit_transform(X_dense)
    colors_palette = ["#ef476f", "#ffd166", "#06d6a0", "#118ab2", "#073b4c", "#073b11", "#073b66", "#073bbb", "#073bbb", "#013bbb", "#063bbb", "#0b3bbb"]
    colors_use = [colors_palette[i] for i in labels]
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=colors_use)
    plt.show()


# initialize the labels
labels = labels_create()


def top_feats_per_cluster(X, y, features, min_tfidf=0.1, top_n=25):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y == label)
        feats_df = top_frq_means(X, features, ids, min_tfidf=min_tfidf, view_num=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def plot_classfeats(dfs):
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i + 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("cluster = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax.barh(x, df.score, align='center', color='#7530FF')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1] + 1])
        yticks = ax.set_yticklabels(df.features)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


# plot_classfeats(top_feats_per_cluster(X, labels, features, 0.1, 25))

