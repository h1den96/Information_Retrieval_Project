from search_engine import *

# Calculate ground truth with usage of BM25 ranking scores
def ground_truth(articles, title_mapping, queries):
    data = []

    for query_id, query_text in queries.items():
        rankings, scores = main_loop(articles, title_mapping, query_text, use='1', method='4')
        
        query_data = []
        for doc_id, score in zip(rankings, scores):
            if score < 10:
                relevance = 0
            elif 10 <= score < 30:
                relevance = 1
            else:
                relevance = 2
            query_data.append((query_id, doc_id, relevance, score))

        data.extend(query_data)
            
    with open("./CISI.REL", "w") as f:
        for row in data:
            f.write("{:5d} {:5d} {:1d} {:10.6f}\n".format(row[0], row[1], row[2], row[3]))

    return None


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


if __name__ == "__main__":

    articles_file = './processed_CISI_articles.json'
    title_articles_file = './CISI_articles.json'
    cisi_queries_path = "./CISI.QRY"
    relevance_file = "./CISI.REL"
    
    articles, title_mapping = load_articles(articles_file), load_titles(title_articles_file)
    
    queries = load_queries(cisi_queries_path)

    if articles and title_mapping:
        ground_truth(articles, title_mapping, queries)
    else:
        print("Failed to load articles or titles. Exiting...")