from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as pst
import spacy

# initialize language for spaCy
sp = spacy.load("en_core_web_sm")

# will get email from a text file
f = open("emails/email1.txt", "r")
email = f.read()

# print("tokenized : ", word_tokenize(email))

# email_tokenized = word_tokenize(email)
# word tokenization works perfectly for this project

# The next phase is to get rid of stopwords which are not giving any useful informations
# the stop words are those :
stopwords_list = set(stopwords.words("english"))
# print(stopwords_list)

# created a new tokenized email list which does not contain stopwords.
# email_nostopwords = [word for word in email_tokenized if not word in stopwords_list]

# print("no stop words : ", ' '.join(email_nostopwords))

# print("Length before :", len(email_tokenized), "vs Length after :", len(email_nostopwords))

# next step is to stem or lemmatize the tokens in order
# to reduce data number to be analysed and hence better performance

# selecting the best tool for this case
# test_words = ['going', 'go', 'went', 'goes', 'gone', 'cat', 'cats', 'kitten']
# print("testing for Porter stemming : ")
# porterRes = []
# for word in test_words:
    # porterRes.append(pst().stem(word))
# print(porterRes)

# print("testing for spaCy : ")
# spacyRes = []
# for word in test_words:
    # spacyRes.extend([token.lemma_ for token in sp(word)])
# print(spacyRes)

# the results shows that spaCy is a better choice in this case as it returns more logical results

# email_string = ' '.join(email_nostopwords)
# email_filtered = [token.lemma_ for token in sp(email_string)]
# print('result after spaCy : ')
# print(email_filtered)

def email_body_filtering(email):
    email_tokenized = word_tokenize(email)
    email_nostopwords = [word for word in email_tokenized if not word in stopwords_list]
    email_string = ' '.join(email_nostopwords)
    email_filtered = [token.lemma_ for token in sp(email_string)]
    print(email_filtered)
    return email_filtered


email_body_filtering(email)