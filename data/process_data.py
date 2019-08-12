import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Loading in the csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merging the DataFrames
    df = pd.merge(messages,categories,on='id')

    return df

def clean_data(df):

    # Creating a separate DataFrame to get categories
    categories = df['categories'].str.split(';',expand=True,)
    
    # Getting and setting column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Getting numbers from data and converting to numeric Dtype
    for column in categories:
        # Setting each value to be the last character / the number
        categories[column] = categories[column].str[-1]
        # Converting column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Dropping original category col
    df = df.drop('categories',axis=1)

    # Concatenating the categories that were extracted above
    df = pd.concat([df,categories],axis=1)

    # Dropping duplicate rows
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    # Creating db engine
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()