"""This module contains functions that will be used in other modules
in order to optimize the performance by not calling a whole module just for one small function.
"""


from nltk.tokenize import word_tokenize
import spacy

sp = spacy.load("en_core_web_sm")


def email_body_filtering(email):
    email_tokenized = word_tokenize(email)
    email_string = ' '.join(email_tokenized)
    email_filtered = ' '.join([token.lemma_ for token in sp(email_string)])

    return email_filtered


def newParse(emails_body, emails_to, emails_from_):
    email_body_filtered = (email_body_filtering(email) for email in emails_body)
    emails_new = {
        'body': (body for body in email_body_filtered),
        'to': (to for to in emails_to),
        'from_': (from_ for from_ in emails_from_)
    }
    return emails_new
