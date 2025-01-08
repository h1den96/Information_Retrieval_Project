import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict
import re
import json

from inverted_index import *

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
logical_operators = {"AND", "OR", "NOT"}

def process_query(text):
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = word_tokenize(cleaned_text)
    
    filtered_stop_words = stop_words - logical_operators
    
    filtered_tokens = [word for word in tokens if word.upper() not in filtered_stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    return lemmatized_tokens
    

def load_articles(json_file):
    try:
        with open(json_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File '{json_file}' is not a valid JSON.")
        return []
    

def query_padding(query):
    padded_query = []
    last_term = False 
    
    for token in query:
        token_upper = token.upper()

        if token_upper in logical_operators:
            if token_upper in {"AND", "OR"} and (not padded_query or not last_term):
                print(f"Error: Invalid query structure near '{token}'.")
                return []

            if token_upper == "NOT":
                padded_query.append(token_upper)
                last_term = False
            elif token_upper in {"AND", "OR"}:
                padded_query.append(token_upper)
                last_term = False
        else:
            if last_term:
                padded_query.append("AND")
            padded_query.append(token_upper)
            last_term = True 

    if not last_term:
        print("Error: Query cannot end with a logical operator.")
        return []

    return padded_query


def main_loop(articles):
    while True:
        #print("\nMenu:")
        #print("1. Make a search")
        #print("2. Exit")
        #choice = input("Enter your choice: ").strip()
        
        #if choice == '1':
        query = input("Enter your query: ").strip()
        processed_query = process_query(query)
        print(f"Processed Query: {processed_query}")

        # Add padding to query
        padded_query = query_padding(processed_query)
        print(f"Padded Query: {padded_query}")

        
#
#            matching_articles = set()
#            for token in processed_query:
#                matching_articles.update(searchIndex(token))
#
#            print("\nSearch Results:")
#            if matching_articles:
#                for article in articles:
#                    if article['id'] in matching_articles:
#                        print(f"\nArticle ID: {article['id']}")
#                        print(f"Tokens: {article['tokens'][:20]}...")
#            else:
#                print("No matching articles found.")
#        
#        elif choice == '2':
#            print("Exiting program.")
#            break
#        else:
#            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    articles_file = './processed_articles.json'
    articles_data = load_articles(articles_file)
    
    if not articles_data:
        print("No articles available for search. Exiting.")
    else:
        main_loop(articles_data)