import pandas as pd

"""First i sliced the csv data into a 5000 email chunk only
 because the original number is a lot for the memory to work with.
"""


def chunking_emails():

    file_path = '/Users/macbookpro/Desktop/INNObyte/modules/emails/emails.csv'
    emails_chunked = pd.read_csv(file_path, nrows=5000)
    # print(emails_chunked)
    return emails_chunked


emails_cut = chunking_emails()


# this function will extract the keys we will work with which are 'body' 'to' 'from'

def parser_email(emails_cut):
    lines = emails_cut.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']

    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message

        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val

    return email


def parse_into_emails(messages):
    emails = [parser_email(message) for message in messages]

    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from_': map_to_list(emails, 'from')
    }


# this function will be used in parse_into_emails to make it add key_value pairs for each key existing in email

def map_to_list(emails, key):
    results = []
    for email in emails:

        if key not in email:
            results.append('')
        else:
            results.append(email[key])

    return results


email_df = pd.DataFrame(parse_into_emails(emails_cut.message))

# now the data is filtered to 4818 email instead of 5000 after
email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)

"""Functions below are getters for emails body, sender
and receiver data.
"""

def emails_df_body():
    return email_df.body


def emails_df_to():
    return email_df.to


def emails_df_from_():
    return email_df.from_
