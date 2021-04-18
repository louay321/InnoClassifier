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

# View related emails index and their content
def view_related_emails(keyword):
    # Transform keyword to vector
    vec_query = vec.transform([keyword])
    cosine_sim = linear_kernel(vec_query, vec_train).flatten()
    # get 100 most related emails
    related_email_indices = cosine_sim.argsort()[:-100:-1]
    print("number of related emails to word :", keyword, " ", related_email_indices)
    for index in related_email_indices:
        print("index : ", index)
        print(" related email : ", df.body[index])
    return 0


# add top 100 related emails to a list of each type
def add_emails_related(keywords):
    arr = []
    for keyword in keywords:
        vec_query = vec.transform([keyword])
        cosine_sim = linear_kernel(vec_query, vec_train).flatten()
        related_email_indices = cosine_sim.argsort()[:-100:-1]
        for index in related_email_indices:
            arr.append(df.body[index])
    return arr


# Here i created a bag of words which contains words of similar context that emails belong to for making categories
notifying = ['receive', 'update', 'confirm', 'receipt', 'inform', 'notify', 'inform']
issue = ['bad', "can 't", 'as issue', 'mistake', 'fault', 'problem', 'worried', 'complain']
request = ['please', 'can you', 'request', 'demand', 'would like']
meeting = ['meeting', 'conference', 'invite', 'arrange', 'call']

# create lists of emails by category
notifying_emails = add_emails_related(notifying)
issue_emails = add_emails_related(issue)
request_emails = add_emails_related(request)
meeting_emails = add_emails_related(meeting)

print(len(notifying_emails))
print(len(issue_emails))
print(request_emails[1])
print(meeting_emails[1])
# removing duplicates

# enumerating the categories
Categories = {'notifying': 0, 'issue': 1, 'request': 2, 'meeting': 3}

# Next step is to create a new dataframe of around 200 emails Categorized after getting rid of duplicates.
data = {
    'Emails content': []
}
# by assigning each email content by its index to the category and creating a new list for dataframe
