
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
def tokenize_stem(text):
    tokens = word_tokenize(text.lower())
    # Removing Stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    # Stemming
    stemmed = [PorterStemmer().stem(w) for w in tokens]
    
    return stemmed