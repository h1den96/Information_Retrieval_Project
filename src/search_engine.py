import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import numpy as np
from collections import defaultdict
import math

# -----------------------------------------------------
# Αρχικοποίηση πόρων NLTK
# -----------------------------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# -----------------------------------------------------
# Global Inverted Index (Ανεστραμμένο Ευρετήριο)
# -----------------------------------------------------
_inverted_index = None

def buildInvertedIndex():
    """
    Δημιουργεί ένα ανεστραμμένο ευρετήριο (inverted index)
    από το αρχείο processed_articles.json.

    Παράδειγμα δομής του processed_articles.json:
    [
      {
        "id": 1,
        "tokens": ["inform", "retriev", "comput", ...]
      },
      ...
    ]
    """
    global _inverted_index
    if _inverted_index is None:
        inverted_index = defaultdict(list)
        try:
            with open('./processed_articles.json', 'r', encoding='utf-8') as file:
                processed_articles = json.load(file)
        except FileNotFoundError:
            print("Σφάλμα: Δεν βρέθηκε το αρχείο processed_articles.json.")
            return {}

        for article in processed_articles:
            for token in set(article["tokens"]):
                inverted_index[token].append(article["id"])
        
        _inverted_index = inverted_index

    return _inverted_index

def searchIndex(term=""):
    """
    Επιστρέφει το σύνολο των doc IDs όπου εμφανίζεται ο term,
    κάνοντας lookup στο inverted index.
    """
    inverted_index = buildInvertedIndex()
    if not inverted_index:
        return set()
    return set(inverted_index.get(term, []))

# -----------------------------------------------------
# Συνάρτηση προεπεξεργασίας ερωτήματος
# -----------------------------------------------------
def process_query(text):
    """
    1. Καθαρισμός κειμένου με regex (αφαίρεση μη αλφαβητικών χαρακτήρων).
    2. Tokenization με nltk.
    3. Αφαίρεση stopwords.
    4. Stemming.
    5. Lemmatization.

    Επιστρέφει λίστα με τα τελικά tokens.
    """
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = word_tokenize(cleaned_text.lower())

    filtered_tokens = [w for w in tokens if w not in stop_words]
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in stemmed_tokens]

    return lemmatized_tokens

# -----------------------------------------------------
# Boolean Search
# -----------------------------------------------------
def evaluate_expression(expression):
    """
    Διαβάζει μια Boolean έκφραση (π.χ. "cat AND dog NOT fish")
    και επιστρέφει σύνολο από doc IDs που την ικανοποιούν.

    Υποστηριζόμενοι τελεστές: AND, OR, NOT
    """
    stack = []
    tokens = expression.split()

    for token in tokens:
        token_up = token.upper()

        if token_up in {"AND", "OR", "NOT"}:
            stack.append(token_up)
        else:
            matching_ids = searchIndex(token)
            stack.append(matching_ids)

        # Προσπαθούμε να κάνουμε μείωση της έκφρασης
        # κάθε φορά που συναντούμε μοτίβο [set, operator, set].
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
        raise ValueError("Μη έγκυρη έκφραση Boolean")

    return stack[0]

def boolean_search(query):
    """
    Ανάλυση μιας Boolean έκφρασης.
    Αν αποτύχει, εφαρμόζουμε fallback σε απλή αναζήτηση (OR) των tokens.
    """
    query = query.strip()
    try:
        return evaluate_expression(query)
    except ValueError:
        # Απλή αναζήτηση OR όλων των tokens
        processed_tokens = process_query(query)
        results = set()
        for token in processed_tokens:
            results.update(searchIndex(token))
        return results

# -----------------------------------------------------
# TF-IDF (με dot product)
# -----------------------------------------------------
def rank_tfidf(query_tokens, articles):
    """
    Υπολογίζει TF-IDF για όλα τα άρθρα, 
    έπειτα κάνει dot product μεταξύ query_vector και κάθε doc_vector.

    Επιστρέφει: (ranked_indices, scores).
    """
    vectorizer = TfidfVectorizer()
    corpus = [" ".join(article['tokens']) for article in articles]
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    query_vector = vectorizer.transform([" ".join(query_tokens)])
    
    scores = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()

    ranked_indices = np.argsort(-scores)
    return ranked_indices, scores

# -----------------------------------------------------
# Vector Space Model (TF-IDF + Cosine Similarity, sklearn)
# -----------------------------------------------------
def rank_vsm(query_tokens, articles):
    """
    Χρησιμοποιεί TfidfVectorizer για όλα τα έγγραφα (corpus) και το query,
    έπειτα υπολογίζει cosine similarity (αντί για απλό dot product).

    Επιστρέφει: (ranked_indices, scores).
    """
    vectorizer = TfidfVectorizer()
    corpus = [" ".join(article['tokens']) for article in articles]
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([" ".join(query_tokens)])

    cos_sims = cosine_similarity(query_vector, tfidf_matrix).flatten()

    ranked_indices = np.argsort(-cos_sims)
    return ranked_indices, cos_sims

# -----------------------------------------------------
# BM25 (χειροποίητη υλοποίηση)
# -----------------------------------------------------
def calc_idf(articles):
    """
    Προϋπολογισμός IDF για κάθε token,
    βάση του οποίου θα υλοποιήσουμε το BM25.
    """
    N = len(articles)
    term_doc_count = defaultdict(int)

    for article in articles:
        unique_tokens = set(article['tokens'])
        for token in unique_tokens:
            term_doc_count[token] += 1

    idf = {}
    for term, doc_count in term_doc_count.items():
        # Τύπος IDF για BM25
        idf[term] = math.log((N - doc_count + 0.5) / (doc_count + 0.5) + 1)
    return idf

def BM25(query_tokens, articles, idf, k1=1.5, b=0.75):
    N = len(articles)
    avg_doc_len = sum(len(a['tokens']) for a in articles) / N

    scores = []
    for article in articles:
        doc_len = len(article['tokens'])
        score = 0
        token_freq = defaultdict(int)

        for token in article['tokens']:
            token_freq[token] += 1

        for term in query_tokens:
            if term in idf:
                tf = token_freq[term]
                idf_value = idf[term]

                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                score += idf_value * (numerator / denominator)

        scores.append(score)

    scores = np.array(scores)
    ranked_indices = np.argsort(-scores)
    return ranked_indices, scores


# -----------------------------------------------------
# Φόρτωση δεδομένων από αρχεία JSON
# -----------------------------------------------------
def load_articles(json_file):
    """
    Φορτώνει τα επεξεργασμένα άρθρα:
    [
      { "id": ..., "tokens": [...] },
      ...
    ]
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Σφάλμα: Το αρχείο '{json_file}' δεν βρέθηκε.")
        return []
    except json.JSONDecodeError:
        print(f"Σφάλμα: Μη έγκυρο JSON στο '{json_file}'.")
        return []

def load_titles(json_file):
    """
    Φορτώνει [{"id": ..., "title": ...}] και επιστρέφει dict {id: title}.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            articles = json.load(file)
        return {article['id']: article['title'] for article in articles}
    except FileNotFoundError:
        print(f"Σφάλμα: Το αρχείο '{json_file}' δεν βρέθηκε.")
        return {}
    except json.JSONDecodeError:
        print(f"Σφάλμα: Μη έγκυρο JSON στο '{json_file}'.")
        return {}

# -----------------------------------------------------
# Συνάρτηση δρομολόγησης (ranking)
# -----------------------------------------------------
def ranking(articles, processed_query, method):
    """
    Επιλέγει έναν από τους 4 διαθέσιμους τρόπους ανάκτησης:

      1. Boolean
      2. TF-IDF (dot product)
      3. BM25
      4. Vector Space Model (TF-IDF + Cosine Similarity)

    Επιστρέφει (doc_ids, scores).
      - Για Boolean, το scores θα είναι κενό ([]) 
        επειδή δεν υπάρχει βαθμολογία (true/false).
    """
    if method == '1':
        # Boolean
        matching_ids = boolean_search(" ".join(processed_query))
        doc_ids = list(matching_ids)
        return doc_ids, []

    elif method == '2':
        # TF-IDF (dot product)
        ranked_indices, scores = rank_tfidf(processed_query, articles)
        doc_ids = [articles[i]['id'] for i in ranked_indices]
        return doc_ids, scores

    elif method == '3':
        # BM25
        idf = calc_idf(articles)
        ranked_indices, scores = BM25(processed_query, articles, idf)
        doc_ids = [articles[i]['id'] for i in ranked_indices]
        return doc_ids, scores

    elif method == '4':
        # Vector Space Model (TF-IDF + Cosine Similarity, sklearn)
        ranked_indices, scores = rank_vsm(processed_query, articles)
        doc_ids = [articles[i]['id'] for i in ranked_indices]
        return doc_ids, scores

    else:
        print("Μη έγκυρη επιλογή αλγορίθμου ανάκτησης.")
        return [], []

# -----------------------------------------------------
# Εμφάνιση αποτελεσμάτων
# -----------------------------------------------------
def display_results(doc_ids, scores, articles, title_mapping):
    """
    Εκτυπώνει τα κορυφαία 10 αποτελέσματα. 
    Αν 'scores' είναι κενό, σημαίνει Boolean mode.
    Διαφορετικά, εκτυπώνει και τη βαθμολογία του κάθε εγγράφου.
    """
    print("\Αποτελέσματα Αναζήτησης:")

    if len(doc_ids) == 0:
        print("Δεν βρέθηκαν αποτελέσματα.")
        return

    top_k = min(10, len(doc_ids))

    if len(scores) == 0:
        # Boolean mode
        for i in range(top_k):
            doc_id = doc_ids[i]
            article_title = title_mapping.get(doc_id, "Χωρίς τίτλο")
            print(f"\nDoc ID: {doc_id}")
            print(f"Τίτλος: {article_title}")
            print("Βαθμολογία: (Boolean - N/A)")
    else:
        # TF-IDF / BM25 / VSM με βαθμολογία
        doc_scores = list(zip(doc_ids, scores))

        for i in range(top_k):
            doc_id, doc_score = doc_scores[i]
            article_title = title_mapping.get(doc_id, "Χωρίς τίτλο")
            print(f"\nRank: {i+1}")
            print(f"Doc ID: {doc_id}")
            print(f"Τίτλος: {article_title}")
            print(f"Βαθμολογία: {doc_score:.4f}")

# -----------------------------------------------------
# Κύριος βρόχος προγράμματος (main_loop)
# -----------------------------------------------------
def main_loop(articles, title_mapping, query=None, use='0', method=None):

    while True:
        if use == '0':
            # Διαδραστικό μενού
            print("\nΜενού:")
            print("1. Αναζήτηση")
            print("2. Έξοδος")
            choice = input("Επιλογή: ").strip()

            if choice == '1':
                user_query = input("Δώστε το ερώτημα (μπορείτε να χρησιμοποιήσετε Boolean operators): ").strip()
                processed_query = process_query(user_query)
                
                print("\nΔιαθέσιμες μέθοδοι:")
                print("1. Boolean Search")
                print("2. TF-IDF (dot product)")
                print("3. Okapi BM25")
                print("4. Vector Space Model (TF-IDF + Cosine Similarity)")
                user_method = input("Επιλογή (1/2/3/4): ").strip()

                doc_ids, scores = ranking(articles, processed_query, user_method)
                display_results(doc_ids, scores, articles, title_mapping)

            elif choice == '2':
                print("Τερματισμός προγράμματος.")
                break
            else:
                print("Μη έγκυρη εντολή. Προσπαθήστε ξανά.")

        elif use == '1':
            if not query:
                print("Δεν δόθηκε ερώτημα. Έξοδος.")
                return [], []

            processed_query = process_query(query)
            doc_ids, scores = ranking(articles, processed_query, method)

            display_results(doc_ids, scores, articles, title_mapping)

            return doc_ids, scores

        else:
            print("Μη έγκυρη λειτουργία (use). Τερματισμός.")
            break

# -----------------------------------------------------
# Παράδειγμα κύριας χρήσης (main)
# -----------------------------------------------------
if __name__ == "__main__":
    articles_file = './processed_articles.json'
    articles = load_articles(articles_file)

    titles_file = './wikipedia_articles.json'
    title_mapping = load_titles(titles_file)

    main_loop(articles, title_mapping, use='0')
