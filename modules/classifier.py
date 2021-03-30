from nltk.tokenize import sent_tokenize, word_tokenize

email = "Dear Mr. John, this email is serialized and has to be tokenized in order to test with it. we'll use words that are hard to tokenize, thank you."

print(word_tokenize(email))

#word tokenization works perfectly for this project