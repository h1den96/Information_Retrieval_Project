from search_engine import main_loop, load_articles, load_titles

def eval_search_engine(queries, ground_truth, articles, title_mapping):

    print("\nSelect Ranking Method for Evaluation:")
    print("1) Boolean Search")
    print("2) TF-IDF (dot product)")
    print("3) Okapi BM25")
    print("4) Vector Space Model (TF-IDF + Cosine Similarity)")
    method_choice = input("Enter your choice (1/2/3/4): ").strip()

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for query_id, query_text in queries.items():
        print(f"\n\nEvaluating Query ID {query_id}: {query_text}")

        # Pass the user-selected method to main_loop
        ranked_indices, scores = main_loop(
            articles, 
            title_mapping, 
            query=query_text, 
            use='1', 
            method=method_choice
        )
        
        retrieved_docs = set(ranked_indices)

        relevant_docs = set(ground_truth.get(query_id, []))
        
        true_positives = len(retrieved_docs & relevant_docs)
        false_positives = len(retrieved_docs - relevant_docs)
        false_negatives = len(relevant_docs - retrieved_docs)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    print("\nOverall Performance:")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1-Score: {avg_f1:.2f}")


def parse_relevance(file_path):
    """
    Parses a relevance file (e.g., CISI.REL) and returns
    a dictionary { query_id: [doc_ids_that_are_relevant] }.
    """
    relevance_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = int(parts[0])
                doc_id = int(parts[1])
                relevance = int(parts[2])
                
                if query_id not in relevance_dict:
                    relevance_dict[query_id] = []

                # If relevance > 0, we consider the doc relevant
                if relevance > 0:
                    relevance_dict[query_id].append(doc_id)
    return relevance_dict


def load_queries(file_path):
    """
    Loads queries from a file like CISI.QRY that has format:
      .I 1
      .W
      <query text here>
      .I 2
      .W
      <query text here>
      etc.

    Returns { query_id: "full query text" }.
    """
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


if __name__ == "__main__":
    articles_file = './processed_CISI_articles.json'
    title_articles_file = './CISI_articles.json'
    cisi_queries_path = "./CISI.QRY"
    relevance_file = "./CISI.REL"
    
    articles = load_articles(articles_file)
    title_mapping = load_titles(title_articles_file)
    queries = load_queries(cisi_queries_path)
    ground_truth = parse_relevance(relevance_file)
    
    if articles and title_mapping:
        eval_search_engine(queries, ground_truth, articles, title_mapping)
    else:
        print("Failed to load articles or titles. Exiting...")
