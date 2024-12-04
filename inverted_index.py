from collections import defaultdict
import json

# Cache the inverted index to avoid rebuilding it on each call
_inverted_index = None

def buildInvertedIndex():
    """Build and cache the inverted index."""
    global _inverted_index
    if _inverted_index is None:  # Build index only once
        inverted_index = defaultdict(list)
        with open('./processed_articles.json', 'r') as file:
            processed_articles = json.load(file)

        for article in processed_articles:
            for token in set(article["tokens"]):  # Use set to avoid duplicates
                inverted_index[token].append(article["id"])
        
        _inverted_index = inverted_index
    return _inverted_index

def searchIndex(term=""):
    """Search for a term in the inverted index. Returns the full index if term is empty."""
    inverted_index = buildInvertedIndex()  # Get the cached index
    if term == "":  # Return the full index if the term is empty
        return inverted_index
    else:  # Return documents that contain the search term
        return inverted_index.get(term, [])  # Returns an empty list if the term is not found