from search_engine import *
from CISI_to_json import *


def eval_search_engine(queries, ground_truth):
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for query_id, query_text in queries.items():
        print(f"Evaluating Query ID {query_id}: {query_text}")
        
        # Retrieve results using your search engine
        retrieved_docs = main_loop(query_text)  # Replace with your search function
        
        # Get relevant docs from ground truth
        relevant_docs = set(ground_truth.get(query_id, []))
        
        # Evaluate
        true_positives = len(retrieved_docs & relevant_docs)
        false_positives = len(retrieved_docs - relevant_docs)
        false_negatives = len(relevant_docs - retrieved_docs)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")


def parse_relevance(file_path):
    relevance_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4:
                query_id = int(parts[0])
                doc_id = int(parts[2])
                if query_id not in relevance_dict:
                    relevance_dict[query_id] = []
                relevance_dict[query_id].append(doc_id)
    return relevance_dict


def load_queries(file_path):
    queries = {}
    current_id = None
    query_text = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('.I'):  # Query ID
                if current_id is not None:  # Save the previous query
                    queries[current_id] = " ".join(query_text).strip()
                current_id = int(line.split()[1])
                query_text = []  # Reset for the next query
            elif line.startswith('.W'):  # Begin query text
                continue
            else:  # Collect query text
                query_text.append(line)
        
        # Save the last query
        if current_id is not None:
            queries[current_id] = " ".join(query_text).strip()

    return queries


if __name__ == "__main__":
    #processed_articles_file = './processed_articles.json'
    articles_file = './processed_CISI.json'
    #articles_data = load_articles(processed_articles)
    #title_mapping = load_titles(articles_file)

    cisi_queries_path = "./CISI.QRY"
    queries = load_queries(cisi_queries_path)

    ground_truth = parse_relevance("./CISI.REL")

    # Print loaded queries
    #for query_id, query_text in queries.items():
    #    print(f"Query ID: {query_id}")
    #    print(f"Query Text: {query_text}")
    #    print("-" * 40)

    #if articles_data:
    #    for query in queries:
    #        main_loop(articles_data, title_mapping, '1', query, '3')  # use = 1, query, method = 3 = BM25
    #    #eval_search_engine(queries, ground_truth)  # Uncomment to evaluate
    #else:
    #    print("Failed to load articles or titles. Exiting...")