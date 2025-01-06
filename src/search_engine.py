import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from inverted_index import searchIndex

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

<<<<<<< HEAD
def tokenize_query(text):
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', text)  
    tokens = word_tokenize(cleaned_text.lower())

=======
def process_query(text):
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = word_tokenize(cleaned_text)
>>>>>>> f01483d6257f7a19d9e34a10c8e45a2d8e214472
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    return lemmatized_tokens

def rank_tfidf(query, articles):
    vectorizer = TfidfVectorizer()
    corpus = [" ".join(article['tokens']) for article in articles]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([" ".join(query)])
    
    scores = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()
    ranked_indices = np.argsort(-scores)
    return ranked_indices, scores

def boolean_search(query):
    if " " in query:
        terms = query.split()
        result = set(searchIndex(terms[0].strip()))
        for term in terms[1:]:
            result &= set(searchIndex(term.strip()))
        return result
    elif "AND" in query:
        terms = query.split("AND")
        return set(searchIndex(terms[0].strip())) & set(searchIndex(terms[1].strip()))
    elif "OR" in query:
        terms = query.split("OR")
        return set(searchIndex(terms[0].strip())) | set(searchIndex(terms[1].strip()))
    elif "NOT" in query:
        terms = query.split("NOT")
        return set(searchIndex(terms[0].strip())) - set(searchIndex(terms[1].strip()))
    else:
        return set(searchIndex(query.strip()))

def main_loop(articles):
    while True:
        print("\nMenu:")
        print("1. Make a search")
        print("2. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            query = input("Enter your query (use AND, OR, NOT for Boolean operations): ").strip()
            processed_query = process_query(query)
            print(f"Processed Query: {processed_query}")

            if algo_choice == '1':  # Boolean Search
                if any(op in query for op in ["AND", "OR", "NOT", " "]):
                    matching_articles = boolean_search(query)
                else:
                    matching_articles = set()
                    for token in processed_query:
                        matching_articles.update(searchIndex(token))

                print("\nSearch Results:")
                if matching_articles:
                    for article in articles:
                        if article['id'] in matching_articles:
                            article_title = title_mapping.get(article['id'], 'No Title')
                            print(f"\nArticle ID: {article['id']}")
                            print(f"Title: {article_title}")
                            print(f"Tokens: {article['tokens'][:20]}...")
                else:
                    print("No matching articles found.")

            elif algo_choice == '2':  # TF-IDF Ranking
                ranked_indices, scores = rank_tfidf(processed_query, articles)
                display_results(ranked_indices, scores, articles, title_mapping)

            else:
                print("Invalid retrieval algorithm choice.")

        elif choice == '2':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

<<<<<<< HEAD
=======
def load_articles(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File '{json_file}' is not a valid JSON.")
        return []
>>>>>>> f01483d6257f7a19d9e34a10c8e45a2d8e214472

def load_titles(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            articles = json.load(file)
        return {article['id']: article['title'] for article in articles}
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File '{json_file}' is not a valid JSON.")
        return {}

if __name__ == "__main__":
    articles_file = './processed_articles.json'
    wikipedia_titles_file = './wikipedia_articles.json'

    articles_data = load_articles(articles_file)
    title_mapping = load_titles(wikipedia_titles_file)

    if not articles_data:
        print("No articles available for search. Exiting.")
    elif not title_mapping:
        print("No titles found for articles. Exiting.")
    else:
        main_loop(articles_data, title_mapping)
