import json
import plotly
import pandas as pd
import plotly.graph_objs as gobj

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine
from models.train_classifier import tokenize_stem

app = Flask(__name__)

# def tokenize_stem(text):
#     tokens = word_tokenize(text.lower())
#     # Removing Stopwords
#     tokens = [w for w in tokens if w not in stopwords.words('english')]
#     # Stemming
#     stemmed = [PorterStemmer().stem(w) for w in tokens]
    
#     return stemmed


# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)
# load model
model = load("./models/model1.joblib")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Data used for plot 1
    melted_df = df.melt(id_vars=['id','message','original','genre'],var_name='category',value_name='value')
    direct_melted_df = melted_df[melted_df['genre']=='direct']
    news_melted_df = melted_df[melted_df['genre']=='news']
    social_melted_df = melted_df[melted_df['genre']=='social']   
    
    # Data for Plot 2
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graphs = [
        {
            'data': [
                Bar(
                    x=[s.replace('_', ' ').title() for s in news_melted_df.groupby('category')['value'].sum().index],
                    y=direct_melted_df.groupby('category')['value'].sum(),
                    name='Direct',
                    marker=dict(color='turquoise')
                ),
                Bar(
                    x=[s.replace('_', ' ').title() for s in news_melted_df.groupby('category')['value'].sum().index],
                    y=news_melted_df.groupby('category')['value'].sum(),
                    name='News',
                    marker=dict(color='tomato')
                ),
                Bar(
                    x=[s.replace('_', ' ').title() for s in social_melted_df.groupby('category')['value'].sum().index],
                    y=social_melted_df.groupby('category')['value'].sum(),
                    name='Social',
                    marker=dict(color='olive')
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35,
                    'automargin': True,
                    
                },
                'barmode': 'stack',
            }
        },
        {
            'data': [
                gobj.Pie(
                    labels=[s.title() for s in genre_names],
                    values=genre_counts,
                    marker=dict(colors=['turquoise','tomato','olive'])
                )
            ],

            'layout': {
                'title': 'Proportions of Message Genres',
            }
        },
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    model_input = pd.DataFrame([[query,'direct']],columns=['message','genre'])

    # use model to predict classification for query
    classification_labels = model.predict(model_input)[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True) # <--- Removed for deployment to web
    # pass


if __name__ == '__main__':
    main()