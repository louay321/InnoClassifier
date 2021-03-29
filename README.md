# INNOclassifier
 An email classifier that is able to identify each each type of email and puts them in classes according to their content.

This project uses NLP(Natural Language Processing) and ML(machine learning) algorithms in order to achieve the required task.
Wrote in python.
 Libraries used are :
  - nltk
  - pandas
  - sciKit
  - (Other stemming and lemmatization libraries might be added later for the Hungarian language)

### System's architecture:
 The project is made up of 4 main modules :
  - Emails fetching from service on the API call.
  - determining the emails type and returning it in a list of tuples -> [(email,type),(email,type)...].
  - Extracting data from the email.
  - Returning the result(email, type, list of extracted data) in a Json format.


