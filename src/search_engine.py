import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json
import numpy as np
from collections import defaultdict

# Text Preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# Inverted Index
_inverted_index = None

def buildInvertedIndex():
    global _inverted_index
    if _inverted_index is None:
        inverted_index = defaultdict(list)
        with open('./processed_articles.json', 'r') as file:
            processed_articles = json.load(file)

        for article in processed_articles:
            for token in set(article["tokens"]):
                inverted_index[token].append(article["id"])
        
        _inverted_index = inverted_index
    return _inverted_index

def searchIndex(term=""):
    inverted_index = buildInvertedIndex()
    return set(inverted_index.get(term, []))


def process_query(text):
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = word_tokenize(cleaned_text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    return lemmatized_tokens

# Boolean Search
def evaluate_expression(expression):
    stack = []
    tokens = expression.split()

    for token in tokens:
        if token.upper() in {"AND", "OR", "NOT"}:
            stack.append(token.upper())
        else:
            stack.append(searchIndex(token))

        while len(stack) >= 3 and isinstance(stack[-1], set) and isinstance(stack[-3], set):
            right = stack.pop()
            operator = stack.pop()
            left = stack.pop()

            if operator == "AND":
                stack.append(left & right)
            elif operator == "OR":
                stack.append(left | right)
            elif operator == "NOT":
                stack.append(left - right)

    if len(stack) != 1 or not isinstance(stack[0], set):
        raise ValueError("Invalid expression")

    return stack[0]

def boolean_search(query):
    query = query.strip()
    try:
        return evaluate_expression(query)
    except ValueError:
        processed_query = process_query(query)
        results = set()
        for token in processed_query:
            results.update(searchIndex(token))
        return results

# TF-IDF Ranking
def rank_tfidf(query, articles):
    vectorizer = TfidfVectorizer()
    corpus = [" ".join(article['tokens']) for article in articles]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([" ".join(query)])
    
    scores = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()
    ranked_indices = np.argsort(-scores)  # Descending order
    return ranked_indices, scores

######

# Calculate IDF for BM25
def calc_idf(articles):
    N = len(articles) 
    term_doc_count = defaultdict(int)

    for article in articles:
        unique_tokens = set(article['tokens'])
        for token in unique_tokens:
            term_doc_count[token] += 1

    idf = {}
    for term, doc_count in term_doc_count.items():
        idf[term] = np.log((N - doc_count + 0.5) / (doc_count + 0.5) + 1)

    return idf

# BM25 Ranking
def BM25(query, articles, idf):

    k1 = 1.5
    b = 0.75

    avg_doc_len = sum(len(article['tokens']) for article in articles) / len(articles)
    scores = []

    for article in articles:
        doc_len = len(article['tokens'])
        score = 0
        token_freq = defaultdict(int)

        for token in article['tokens']:
            token_freq[token] += 1

        for term in query:
            if term in idf:
                tf = token_freq[term]
                idf_term = idf[term]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                score += idf_term * (numerator / denominator)

        scores.append(score)

    ranked_indices = np.argsort(-np.array(scores))
    return ranked_indices, scores

######

# Display Results
def display_results(ranked_indices, scores, articles, title_mapping):
    print("\nRanked Search Results:")
    for idx in ranked_indices[:10]:  # Top 10 results
        article = articles[idx]
        article_title = title_mapping.get(article['id'], 'No Title')
        print(f"\nArticle ID: {article['id']}")
        print(f"Title: {article_title}")
        print(f"Score: {scores[idx]:.4f}")
        print(f"Tokens: {article['tokens'][:20]}...")

# Load Data
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

# Main Program Loop
def main_loop(articles, title_mapping):
    while True:
        print("\nMenu:")
        print("1. Make a search")
        print("2. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            query = input("Enter your query (use AND, OR, NOT for Boolean operations): ").strip()
            processed_query = process_query(query)
            print(f"Processed Query: {processed_query}")

            print("\nChoose retrieval algorithm:")
            print("1. Boolean Search")
            print("2. TF-IDF Ranking")
            print("3. Okapi BM35 Ranking")
            algo_choice = input("Enter your choice: ").strip()

            if algo_choice == '1':  # Boolean Search
                matching_articles = boolean_search(query)

                print("\nSearch Results:")
                if matching_articles:
                    for article in articles:
                        if article['id'] in matching_articles:
                            article_title = title_mapping.get(article['id'], 'No Title')
                            print(f"\nArticle ID: {article['id']}")
                            #print(f"Title: {article_title}")
                            #print(f"Tokens: {article['tokens'][:20]}...")
                else:
                    print("No matching articles found.")

            elif algo_choice == '2':  # TF-IDF Ranking
                ranked_indices, scores = rank_tfidf(processed_query, articles)
                display_results(ranked_indices, scores, articles, title_mapping)

            elif algo_choice == '3':
                idf = calc_idf(articles)  # Precompute IDF
                ranked_indices, scores = BM25(processed_query, articles, idf)
                display_results(ranked_indices, scores, articles, title_mapping)

            else:
                print("Invalid retrieval algorithm choice.")

        elif choice == '2':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    processed_articles_file = './processed_articles.json'
    articles_file = './wikipedia_articles.json'

    articles_data = load_articles(processed_articles_file)
    title_mapping = load_titles(articles_file)

    if not articles_data:
        print("No articles available for search. Exiting.")
    elif not title_mapping:
        print("No titles found for articles. Exiting.")
    else:
        main_loop(articles_data, title_mapping)
