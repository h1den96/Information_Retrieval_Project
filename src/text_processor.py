import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict
import re
import json

# Downloading NLTK data if needed
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load JSON file
with open('./wikipedia_articles.json', 'r') as file:
    articles_data = json.load(file)

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Initialize list to hold tokens for each article
articles_tokens = []

for article in articles_data:
    # Ensure 'content' key exists
    if 'content' in article:        # Might cause problems if there are no contents. Mismatch content-id
        # Step 1: Clean the content
        content_cleaned = re.sub(r'[^A-Za-zΑ\s]', '', article['content'])
        
        # Step 2: Tokenize the cleaned content
        article_tokens = word_tokenize(content_cleaned)
        
        # Step 3: Remove stopwords
        filtered_tokens = [word for word in article_tokens if word.lower() not in stop_words]
        
        # Step 4: Apply stemming (optional for English)
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        
        # Step 5: Apply lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
        
        articles_tokens.append(lemmatized_tokens)

        print(f"Article ID: {article['id']}")
        print(f"Contents: {article['content'][:200]}...")


    processed_articles = [
    {
        "id": article["id"],
        "tokens": lemmatized_tokens
    }
    for article, lemmatized_tokens in zip(articles_data, articles_tokens)
    ]

    with open("processed_articles.json", "w") as f:
        json.dump(processed_articles, f)

