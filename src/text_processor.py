import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict
import re
import json

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

with open('./wikipedia_articles.json', 'r') as file:
    articles_data = json.load(file)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

articles_tokens = []

for article in articles_data:
    if 'content' in article:

        content_cleaned = re.sub(r'[^A-Za-zΑ\s]', '', article['content']) 
        article_tokens = word_tokenize(content_cleaned)
        filtered_tokens = [word for word in article_tokens if word.lower() not in stop_words]
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
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