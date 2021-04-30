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

print('before ', len(notifying_emails))
print('before ', len(issue_emails))
print('before ', len(request_emails))
print('before ', len(meeting_emails))

# removing duplicates
notifying_emails = list(set(notifying_emails))
issue_emails = list(set(issue_emails))
request_emails = list(set(request_emails))
meeting_emails = list(set(meeting_emails))

print('after ', len(notifying_emails))
print('after ', len(issue_emails))
print('after ', len(request_emails))
print('after ', len(meeting_emails))

# enumerating the categories
Categories = {'notifying': 0, 'issue': 1, 'request': 2, 'meeting': 3}

# Next step is to create a new dataframe of around 200 emails Categorized after getting rid of duplicates.
# Xored = set(notifying_emails) ^ set(issue_emails) ^ set(request_emails) ^ set(meeting_emails)
# print('XOR ', len(Xored))
# After XORing the emails to find out how many total unique emails there are it was only 500 emails
# which is not good enough to train the model so i will just work with some duplicated ones

#creates a new dataframe for each emails category
def dataframe_category(category,cat_emails):
    cat_list = []
    for i in cat_emails:
        cat_list.append(category)
    data_list = {
    'body': cat_emails,
    'category': cat_list
    }
    df_category = pd.DataFrame(data_list, columns=['body', 'category'])
    return df_category

# combining emails dataframes in one dataframe
def combine_dataframe():
    df_notifying = dataframe_category('notifying', notifying_emails)
    df_issue = dataframe_category('issue', issue_emails)
    df_request = dataframe_category('request', request_emails)
    df_meeting = dataframe_category('meeting', meeting_emails)

    df_final = df_notifying.append([df_issue, df_request, df_meeting])
    return df_final

df_final = combine_dataframe()
print(df_final)

# df_final.to_csv(r'/Users/macbookpro/Desktop/INNObyte/modules/categorized_emails.csv')
# by assigning each email content by its index to the category and creating a new list for dataframe
