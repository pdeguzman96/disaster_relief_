
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
def tokenize_stem(text):
    '''
    This function tokenizes a string by converting all to lowercase, removing stopwords, and stemming.

    Input
    ----------------
    text: string to be tokenized

    Returns
    ----------------
    List of tokens
    '''
    tokens = word_tokenize(text.lower())
    # Removing Stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    # Stemming
    stemmed = [PorterStemmer().stem(w) for w in tokens]
    
    return stemmed