import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Sample DataFrame with tweets
data = {'tweet': [
    "This is a sample tweet!",
    "It contains some #hashtags and @mentions",
    "Text preprocessing is important for NLP tasks",
    "I love natural language processing!",
    "Machine learning models require clean data",
    "NLTK provides useful tools for text processing",
    "Data cleaning is essential for accurate analysis",
    "Tokenization breaks text into smaller units",
    "Stemming reduces words to their root form",
    "Stopword removal improves text analysis"
]}
df = pd.DataFrame(data)

# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Removing Punctuation and Special Characters
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]
    
    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

# Clean text data
df['clean_text'] = df['tweet'].apply(preprocess_text)

# Tokenize the preprocessed text
tokenized_text = df['clean_text'].apply(word_tokenize)

print("Original Data:")
print(df['tweet'])
print("\nPreprocessed Data:")
print(df['clean_text'])
print("\nTokenized Data:")
print(tokenized_text)
