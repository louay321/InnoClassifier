# This project is a proof of concept for the Email Classifier project assigned by INNObyte

from modules.classifier import *

def main():
    # the first step is to fetch the requested email and return it in a list. this module will be added in the end.
    # email = fetch_email(request)

    # After fetching the email it will analyse it to determine its type and 'classify' it, this part uses NLP.
    # stemming in hungarian library/algorithms to be added
    # response_email = classify(email)

    # The classify() module will return a tuple containing the email and its class

    # The email will be used in the next method as parameter to extract data such as name, address, etc..
    # response_data = extract_data(email)

    # Data rendering in a UI that is custom created OR send data back as Json based etc..
    # response = JSON([response_email, response_data]).
    # return response
    return 1


main()
