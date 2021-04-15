"""In this module i will label and assign a category for each of the emails
depending on results from the classifier.py module.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

from modules.chunks import email_df
from modules.tools import *

# preprocessing the emails and creating a new dataframe df and saving it in a file

path = '/Users/macbookpro/Desktop/INNObyte/modules/emails/'
filename = 'exported_df.csv'

def create_new_df():
    df = pd.DataFrame(newParse(emails_body=email_df.body, emails_to=email_df.to, emails_from_=email_df.from_))
    # export the df after transforming for further testings
    df.to_csv(path + filename, header=True)
    return 0
# Call only when dataframe size changed
# create_new_df()
df = pd.read_csv(path + filename)

# vector creation
stopwords = ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient'])
vec = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)
vec_train = vec.fit_transform(df.body)

# find related emails by using cosine similarity to a specific keyword taken from classifier.py module

keyword = "meeting"

# Transform keyword to vector
vec_query = vec.transform([keyword])

cosine_sim = linear_kernel(vec_query, vec_train).flatten()

# Find top 100 most related emails to the query.
related_email_indices = cosine_sim.argsort()[:-200:-1]
print("related emails to word :", keyword, " ", related_email_indices)

# print out the first email
# first_email_index = related_email_indices[0]

# printing the emails related to that topic
for i in range(10):
    print(df.body.to_numpy()[related_email_indices[i]])

# Here i created a bag of words which contains words of similar context that emails belong to for making a category from each
update = ['receive', 'update', 'confirm', 'receipt']
problem = ['bad', "can 't", 'issue', 'mistake', 'fault']
request = ['please', 'can you', 'request', 'demand']
meeting = ['meeting', 'conference', 'invite']

# enumerating the categories
# Categories = {update: 0, problem: 1, request: 2}
# Next step is to create a new dataframe of around 200 emails Categorized after getting rid of duplicates.
