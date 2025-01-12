import nltk
import re
import json
import sys
import math
import numpy as np

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------
# Αρχικοποίηση NLTK (αν δεν έχει ήδη γίνει)
# ----------------------------------------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

_inverted_index = None

# ---------------------------------------------------------
# Φόρτωση άρθρων, τίτλων κ.λπ.
# ---------------------------------------------------------
def load_articles(json_file):
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

def load_queries(file_path):
    queries = {}
    current_id = None
    query_text = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('.I'):
                if current_id is not None:
                    queries[current_id] = " ".join(query_text).strip()
                current_id = int(line.split()[1])
                query_text = []
            elif line.startswith('.W'):
                continue
            else:
                query_text.append(line)
        if current_id is not None:
            queries[current_id] = " ".join(query_text).strip()
    return queries

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
def process_query(text):
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = word_tokenize(cleaned_text.lower())
    filtered_tokens = [w for w in tokens if w not in stop_words]
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in stemmed_tokens]
    return lemmatized_tokens

# ---------------------------------------------------------
# Inverted Index & Boolean
# ---------------------------------------------------------
def buildInvertedIndex(articles):
    global _inverted_index
    if _inverted_index is None:
        inverted_index = defaultdict(list)
        for article in articles:
            for token in set(article["tokens"]):
                inverted_index[token].append(article["id"])
        _inverted_index = inverted_index
    return _inverted_index

def searchIndex(term, articles):
    inverted_index = buildInvertedIndex(articles)
    return set(inverted_index.get(term, []))

def evaluate_expression(expression, articles):
    stack = []
    tokens = expression.split()
    for token in tokens:
        token_up = token.upper()
        if token_up in {"AND", "OR", "NOT"}:
            stack.append(token_up)
        else:
            matching = searchIndex(token, articles)
            stack.append(matching)
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

    if len(stack) == 1 and isinstance(stack[0], set):
        return stack[0]
    else:
        return set()

def boolean_search(query_text, articles):
    query_text = query_text.strip()
    try:
        result_set = evaluate_expression(query_text, articles)
        if not result_set:
            raise ValueError("Empty result from Boolean")
        return list(result_set), []
    except:
        # fallback OR
        processed = process_query(query_text)
        results = set()
        for t in processed:
            results.update(searchIndex(t, articles))
        return list(results), []

# ---------------------------------------------------------
# TF-IDF (dot product)
# ---------------------------------------------------------
def rank_tfidf(query_tokens, articles):
    vectorizer = TfidfVectorizer()
    corpus = [" ".join(a['tokens']) for a in articles]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([" ".join(query_tokens)])
    scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
    # Normalization σε 0-100
    scores *= 100
    ranked_indices = np.argsort(-scores)
    return ranked_indices, scores

# ---------------------------------------------------------
# TF-IDF + Cosine Similarity
# ---------------------------------------------------------
def rank_vsm(query_tokens, articles):
    vectorizer = TfidfVectorizer()
    corpus = [" ".join(a['tokens']) for a in articles]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([" ".join(query_tokens)])
    cos_sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Normalization σε 0-100
    cos_sims *= 100
    ranked_indices = np.argsort(-cos_sims)
    return ranked_indices, cos_sims

# ---------------------------------------------------------
# BM25
# ---------------------------------------------------------
def calc_idf(articles):
    N = len(articles)
    term_doc_count = defaultdict(int)
    for a in articles:
        unique_tokens = set(a['tokens'])
        for t in unique_tokens:
            term_doc_count[t] += 1
    idf = {}
    for t, doc_count in term_doc_count.items():
        idf[t] = math.log((N - doc_count + 0.5)/(doc_count + 0.5) + 1)
    return idf

def BM25(query_tokens, articles, idf):
    k1 = 1.5
    b = 0.75
    N = len(articles)
    avg_len = sum(len(a['tokens']) for a in articles)/N
    scores = []
    for a in articles:
        freq = defaultdict(int)
        for t in a['tokens']:
            freq[t] += 1
        score = 0
        for q in query_tokens:
            if q in idf:
                tf = freq[q]
                numerator = tf*(k1+1)
                denominator = tf + k1*(1 - b + b*(len(a['tokens'])/avg_len))
                score += idf[q]*(numerator/denominator)
        scores.append(score)
    scores = np.array(scores)
    # Normalization σε 0-100
    scores *= 100
    ranked_indices = np.argsort(-scores)
    return ranked_indices, scores

# ---------------------------------------------------------
# Συνάρτηση Ranking
# ---------------------------------------------------------
def ranking(articles, query_text, method):
    processed_query = process_query(query_text)
    if method == '1':
        # Boolean
        docs, _ = boolean_search(query_text, articles)
        return docs, []
    elif method == '2':
        # TF-IDF
        ranked_indices, sc = rank_tfidf(processed_query, articles)
        doc_ids = [articles[i]['id'] for i in ranked_indices]
        scores_ordered = [sc[i] for i in ranked_indices]
        return doc_ids, scores_ordered
    elif method == '3':
        # BM25
        idf_vals = calc_idf(articles)
        ranked_indices, sc = BM25(processed_query, articles, idf_vals)
        doc_ids = [articles[i]['id'] for i in ranked_indices]
        scores_ordered = [sc[i] for i in ranked_indices]
        return doc_ids, scores_ordered
    elif method == '4':
        # TF-IDF + Cosine
        ranked_indices, sc = rank_vsm(processed_query, articles)
        doc_ids = [articles[i]['id'] for i in ranked_indices]
        scores_ordered = [sc[i] for i in ranked_indices]
        return doc_ids, scores_ordered
    else:
        print("Μη έγκυρη μέθοδος.")
        return [], []

# ---------------------------------------------------------
# main_loop (διαδραστικό ή one-shot)
# ---------------------------------------------------------
def main_loop(articles, title_mapping, query=None, use='0', method=None):
    if use == '1':
        # one-shot
        doc_ids, scores = ranking(articles, query, method)
        return doc_ids, scores
    else:
        while True:
            print("\nMenu:")
            print("1) Search")
            print("2) Exit")
            choice = input("Choice: ").strip()
            if choice == '1':
                user_query = input("Enter your query: ")
                print("Methods:\n1) Boolean\n2) TF-IDF\n3) BM25\n4) TF-IDF+Cosine")
                user_method = input("Select (1..4): ").strip()
                docs, scores = ranking(articles, user_query, user_method)
                top_k = min(10, len(docs))
                for i in range(top_k):
                    did = docs[i]
                    sc = scores[i] if scores else 0
                    title = title_mapping.get(did, "No Title")
                    print(f"{i+1}. DocID={did} Score={sc:.2f} Title={title}")
            elif choice == '2':
                print("Exiting interactive mode.")
                break
        return [], []

# ---------------------------------------------------------
# Ground Truth creation
# ---------------------------------------------------------
def ground_truth(articles, title_mapping, queries, method_for_gt='3'):
    """
    Δημιουργεί/ενημερώνει το αρχείο CISI.REL
    πολλαπλασιάζοντας το score x100 (ήδη το κάνουμε στο ranking()),
    και ορίζοντας relevance=0,1,2 με κάποιον όρο.
    """
    data = []
    for qid, qtext in queries.items():
        ranked_docs, scores = main_loop(articles, title_mapping, qtext, use='1', method=method_for_gt)
        query_data = []
        for doc_id, sc in zip(ranked_docs, scores):
            # Παράδειγμα threshold:
            # score < 10 => rel=0, 10..30 => rel=1, >30 => rel=2
            if sc < 10:
                relevance = 0
            elif 10 <= sc < 30:
                relevance = 1
            else:
                relevance = 2
            query_data.append((qid, doc_id, relevance, sc))
        data.extend(query_data)
    # Γράφουμε σε CISI.REL
    with open("./CISI.REL", "w") as f:
        for row in data:
            f.write("{:5d} {:5d} {:1d} {:10.4f}\n".format(row[0], row[1], row[2], row[3]))
    print("CISI.REL updated successfully.")

# ---------------------------------------------------------
# Parse Relevance
# ---------------------------------------------------------
def parse_relevance(file_path):
    relevance_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid = int(parts[0])
                did = int(parts[1])
                rel = int(parts[2])
                if qid not in relevance_dict:
                    relevance_dict[qid] = []
                # Θεωρούμε relevant όσα έχουν rel>0
                if rel > 0:
                    relevance_dict[qid].append(did)
    return relevance_dict

# ---------------------------------------------------------
# Evaluate Search Engine
# ---------------------------------------------------------
def eval_search_engine(queries, ground_truth_dict, articles, title_mapping):
    print("\nSelect Ranking Method for Evaluation:")
    print("1) Boolean Search")
    print("2) TF-IDF (dot product)")
    print("3) Okapi BM25")
    print("4) TF-IDF + Cosine Similarity")
    method_choice = input("Enter your choice (1/2/3/4): ").strip()

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for qid, qtext in queries.items():
        print(f"\nEvaluating Query ID {qid}: {qtext}")
        doc_ids, _scores = main_loop(articles, title_mapping, qtext, use='1', method=method_choice)
        retrieved_docs = set(doc_ids)
        relevant_docs = set(ground_truth_dict.get(qid, []))

        print(f"Retrieved Docs: {len(retrieved_docs)}, Relevant Docs: {len(relevant_docs)}")

        tp = len(retrieved_docs & relevant_docs)
        fp = len(retrieved_docs - relevant_docs)
        fn = len(relevant_docs - retrieved_docs)

        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    avg_p = sum(precision_scores)/len(precision_scores) if precision_scores else 0
    avg_r = sum(recall_scores)/len(recall_scores) if recall_scores else 0
    avg_f1= sum(f1_scores)/len(f1_scores) if f1_scores else 0

    print("\nOverall Performance:")
    print(f"Avg Precision: {avg_p:.2f}")
    print(f"Avg Recall:    {avg_r:.2f}")
    print(f"Avg F1-Score:  {avg_f1:.2f}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    articles_file = None
    titles_file   = None
    queries_file  = None
    rel_file      = None

    # Δίνουμε επιλογές στον χρήστη:
    print("\nΤι θέλετε να κάνετε;")
    print("1) CISI Evaluation (create/update CISI.REL & evaluate)")
    print("2) Interactive Search (search in CLI)")
    user_choice = input("Επιλογή (1/2): ").strip()

    if user_choice == '1':
        # Ζητάμε μέθοδο για δημιουργία ground truth
        print("\nΜε ποια μέθοδο θα φτιάξουμε το CISI.REL;")
        print("1) Boolean")
        print("2) TF-IDF (dot product)")
        print("3) BM25")
        print("4) TF-IDF + Cosine")
        method_for_gt = input("Επιλογή (1..4): ").strip()
        if method_for_gt not in {'1','2','3','4'}:
            print("Μη έγκυρη, χρήση default BM25 (3).")
            method_for_gt='3'
        
        articles_file = './processed_CISI_articles.json'
        titles_file   = './CISI_articles.json'
        queries_file  = './CISI.QRY'
        rel_file      = './CISI.REL'

        # Φορτώνουμε dataset
        articles = load_articles(articles_file)
        title_mapping = load_titles(titles_file)
        queries = load_queries(queries_file)

        if not (articles and title_mapping and queries):
            print("Missing dataset files. Exiting.")
            sys.exit(1)
        
        # Δημιουργούμε/ενημερώνουμε CISI.REL
        ground_truth(articles, title_mapping, queries, method_for_gt)
        # Φορτώνουμε το ground truth σε λεξικό
        ground_truth_dict = parse_relevance(rel_file)
        # Τρέχουμε eval
        eval_search_engine(queries, ground_truth_dict, articles, title_mapping)

    elif user_choice == '2':
        # Διαδραστική αναζήτηση
        articles_choice = input("Ποιά δεδομένα εισόδου να χρησιμοποιηθούν: 1.Wikipedia 2.CISI (1/2): ").strip()
        if articles_choice == '1':
            articles_file = './processed_articles.json'
            titles_file   = './wikipedia_articles.json'
        elif articles_choice == '2':
            articles_file = './processed_CISI_articles.json'
            titles_file   = './CISI_articles.json'
        else:
            print("Μη έγκυρη επιλογή.")
            sys.exit(1)
        
        articles = load_articles(articles_file)
        title_mapping = load_titles(titles_file)

        if not articles or not title_mapping:
            print("Missing dataset files. Exiting.")
            sys.exit(1)

        main_loop(articles, title_mapping, use='0')

    
    else:
        print("Μη έγκυρη επιλογή. Τερματισμός.")
