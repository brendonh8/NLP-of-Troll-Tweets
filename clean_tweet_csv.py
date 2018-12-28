import pandas as pd
import string
import re
import os
from datetime import datetime


def clean_column_initial(row):
    '''
    Takes a column and cleans all rows of punctuation/hyperlinks etc
    '''
    row = ''.join(x for x in row if not x.isdigit())
    row = re.sub(r'@[A-Za-z0-9]+', '', row)
    row = re.sub('https?://[A-Za-z0-9./]+', '', row)
    row = row.replace(u'\xa0', ' ')
    row = ''.join(x for x in row if x not in string.punctuation)
    row = row.lower()
    return row


alphabet = string.ascii_letters + string.punctuation
# Change to name of column that contains tweets in file
tweet_column = 'content'


def to_datetime(d):
    return datetime.strptime(d, '%m/%d/%Y %H:%M')


def tweet_cleaner():
    '''
    Script runs on a directory containing csv files with raw tweets in a column
    Returns pickle files with cleaned tweet columns
    '''
    file_number = 0
    # Specify directory name
    for (dirname, dirs, files) in os.walk('raw_files'):
        for filename in files:
            file_number += 1
            if filename.endswith('.csv'):
                df_tweets_full = pd.read_csv(os.path.join('raw_files', filename), encoding='utf-8', converters={'publish_date': to_datetime})

            # Use only english tweets
            df_english = df_tweets_full[df_tweets_full.language == 'English']
            # Drop na rows for initial cleaning
            df_english = df_english[pd.notnull(df_english[tweet_column])]
            # Apply cleaning function to English df
            clean_df = df_english[tweet_column].apply(clean_column_initial)
            # Save characters other than the alphabet to be removed
            unwanted_char = []
            for row in clean_df:
                unwanted_char.append(''.join(filter(lambda x: x not in alphabet, row)).replace(' ', ''))
            unwanted_char = list(filter(None, unwanted_char))

            remove_set = set()

            def clean_column_final(row):
                '''
                Takes a column and removes unwanted characters from rows
                '''
                row = ''.join(x for x in row if x not in remove_set)
                return row

            for entry in set(unwanted_char):
                for character in entry:
                    remove_set.add(character)

            clean_df = clean_df.apply(clean_column_final)
            df_english.loc[:, tweet_column] = clean_df.values

            # Remove unneeded columns
            unneeded_col = ['external_author_id', 'harvested_date',
                            'alt_external_id', 'article_url',
                            'tco1_step1', 'tco2_step1', 'tco3_step1']

            df_english.drop(unneeded_col, axis=1, inplace=True)
            # Drop any remaining na rows after cleaning
            df_english = df_english[pd.notnull(df_english[tweet_column])]

            if not os.path.exists('clean_pickles'):
                os.makedirs('clean_pickles')

            df_english.to_pickle('clean_pickles/tweets' + str(file_number) + '.pkl')
            print('Wrote File:', str(file_number))

if __name__ == "__main__":
    tweet_cleaner()
