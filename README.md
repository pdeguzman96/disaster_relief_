# Disaster Relief Pipeline and Interactive Dashboard

This project utilizes an ETL pipeline and ML pipeline to load and preprocess a dataset of labeled disaster-resposne messages then train a multi-output classifier to predict new messages.

You can interact with a trained classifier and look at a couple plots describing the training dataset [here](https://pd-disasterrelief.herokuapp.com/).

## Repo Contents

- data
    - DisasterResponse.db: cleaned and preprocessed data ready to train a multi-output classiier. Output from `process_data.py`.
    - disaster_categories.csv: dataset containing messages' labels/categories
    - disaster_messages.csv: dataset containing messages
    - process_data.py: script used to preprocess files like `disaster_categories.csv` and `disaster_messages.csv`
- templates
    - go.html: extentsion of `master.html` used to modify the webpage with classification results from user's query
    - master.html: landing webpage for the project
- Procfile: used to deploy the Heroku webapp
- classifier.joblib: pickled trained pipeline output from `train_classifier.py`.
- cust_tokenizer.py: containes `tokenize_stem` function used in `train_classifier.py` and `run.py`.
- requirements.txt: contains library versions required to host webapp. Required for Heroku.

## Instructions

In order to 

## Dependencies

- sqlalchemy
- plotly
- pandas
- nltk
- flask
- sklearn
- gunicorn