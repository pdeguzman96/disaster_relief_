import sys
import time
import pandas as pd
import numpy as np
from joblib import dump,load
from sqlalchemy import create_engine
import warnings

# ML
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,make_scorer,f1_score
from sklearn.model_selection import GridSearchCV
from cust_tokenizer import tokenize_stem

def load_data(database_filepath):
    '''
    Loads data given a database filepath using sqlite.

    Input
    ==============
    database_filepath: filepath to the database file of messages

    Returns
    ==============
    X and Y matrix
    '''
    # Creating link to database file
    engine = create_engine('sqlite:///'+database_filepath)
    # Reading in data
    df = pd.read_sql_table('messages',engine)
    # Dropping unnecessary columns
    df = df.drop(['id','original'],axis=1)
    # Creating X and Y matrices
    X = df.iloc[:,:2]
    Y = df.iloc[:,2:]

    return X, Y

def build_model():
    '''
    Creates a full pipeline

    Returns
    ==============
    Pipeline object to be trained
    '''
    # Pipeline to process the messages
    msg_pipeline = Pipeline([
        ('tfidf',TfidfVectorizer(tokenizer=tokenize_stem)),
    ])
    # Pieline to process the genres
    genre_pipeline = Pipeline([
        ('ohe',OneHotEncoder())
    ])
    # Combining these into a column transformer
    combined_pipe = ColumnTransformer([
        ('msg',msg_pipeline,'message'),
        ('genre',genre_pipeline,['genre'])
    ])
    # Full pipeline with clf
    full_pipeline = Pipeline([
        ('preproc',combined_pipe),
        # This was the best classifier after performing a GridSearch within a separate Jupyter Notebook
        ('clf',MultiOutputClassifier(BernoulliNB(alpha=0.3)))
    ])

    return full_pipeline

def custom_scorer(y_true,y_pred):
    '''
    Used to score for GridSearch. Averages f1-score across all 36 categories
    '''
    running_f1 = 0
    for i,col in enumerate(y_true):
        running_f1 += f1_score(y_true[col],y_pred[:,i],average='macro')
    return running_f1/len(y_true.columns)

def evaluate_model(model, X_test, y_test):
    '''
    Calculates and prints metrics for each category, then averages and prints all 36 metrics.
    '''
    # Setting Running Metrics
    running_accuracy=0
    running_weighted_f1=0
    running_weighted_precision=0
    running_weighted_recall=0    
    # Getting predictions
    y_preds = model.predict(X_test)
    # Iterating through each category
    for i,col in enumerate(y_test):
        print(f'Metrics for "{col}":')
        print(classification_report(y_test[col],y_preds[:,i]))
        clf_rpt = classification_report(y_test[col],y_preds[:,i],output_dict=True)
        running_accuracy += clf_rpt['accuracy']
        running_weighted_f1 += clf_rpt['weighted avg']['f1-score']
        running_weighted_precision += clf_rpt['weighted avg']['precision']
        running_weighted_recall += clf_rpt['weighted avg']['recall']
    # Averaging Running Metrics
    avg_acc = running_accuracy/y_test.shape[1]
    avg_wtd_f1 = running_weighted_f1/y_test.shape[1]
    avg_wtd_precision = running_weighted_precision/y_test.shape[1]
    avg_wtd_recall = running_weighted_recall/y_test.shape[1]
    # Printing Running Metrics
    print(f'Average Accuracy: {avg_acc:.4f}')
    print(f'Average F1-Score: {avg_wtd_f1:.4f}')
    print(f'Average Precision: {avg_wtd_precision:.4f}')
    print(f'Average Recall: {avg_wtd_recall:.4f}')
    return None

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        warnings.filterwarnings("ignore")
        parameters = [
              {'clf__estimator':[BernoulliNB()],
               'clf__estimator__alpha':np.arange(0.3,1.7,.25)},
              {'clf__estimator':[RandomForestClassifier()],
               'clf__estimator__n_estimators': np.arange(20,51,10),
               'clf__estimator__criterion':['gini','entropy']}            
             ]
        cust_scorer = make_scorer(custom_scorer)
        grid_search = GridSearchCV(model,parameters,scoring=cust_scorer,cv=3,n_jobs=-1)
        grid_search.fit(X_train,y_train)
        model = grid_search.best_estimator_

        print('Evaluating model & printing classification report for each category...')
        evaluate_model(model, X_test, y_test)

        print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
        dump(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ./data/DisasterResponse.db classifier.joblib')
    
if __name__ == '__main__':
    main()