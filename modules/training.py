import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
import pickle

# read the categorized emails dataset
def get_df():
    file_path = '/Users/macbookpro/Desktop/INNObyte/modules/categorized_emails.csv'
    emails_df = pd.read_csv(file_path)
    return emails_df


emails_df = get_df()

# enumerating the categories
Categories = {'notifying': 0, 'issue': 1, 'request': 2, 'meeting': 3}

def enumerating(emails_df):
    emails_df['Category_Code'] = emails_df['category']
    emails_df = emails_df.replace({'Category_Code': Categories})
    return emails_df

emails_df = enumerating(emails_df)

# splitting the test and train dataset by taking only 15% for testing because the full dataset is not big
X_train, X_test, y_train, y_test = train_test_split(emails_df['body'], emails_df['Category_Code'],
                                                    test_size=0.15, random_state=8)
# param init
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

# here i ran the tfidf feature extraction again in order to get
# more precise features which will be used to train the model
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
# chi2 method to get most correlated unigrams
for Product, category_id in sorted(Categories.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-2:])))

# after printing out the most 2 correlated diagrams it seemed logical so i went with it
# and saved it into pickles

# X_train
with open('Pickles/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)

# X_test
with open('Pickles/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)

# y_train
with open('Pickles/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)

# y_test
with open('Pickles/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)

# df
with open('Pickles/df.pickle', 'wb') as output:
    pickle.dump(emails_df, output)

# features_train
with open('Pickles/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('Pickles/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('Pickles/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('Pickles/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)

# TF-IDF object
with open('Pickles/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)
