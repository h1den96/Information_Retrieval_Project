#from collections import defaultdict
#import json
#
#_inverted_index = None
#
#def buildInvertedIndex():
#    global _inverted_index
#    if _inverted_index is None:
#        inverted_index = defaultdict(list)
#        with open('./processed_articles.json', 'r') as file:
#            processed_articles = json.load(file)
#
#        for article in processed_articles:
#            for token in set(article["tokens"]):
#                inverted_index[token].append(article["id"])
#        
#        _inverted_index = inverted_index
#    return _inverted_index
#
#def searchIndex(term=""):
#    inverted_index = buildInvertedIndex()
#    if term == "":
#        return inverted_index
#    else:
#        return inverted_index.get(term, [])`


from collections import defaultdict
import json

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
    if term == "":
        return list(inverted_index.keys())  # Return only terms (if desired).
    else:
        return inverted_index.get(term, [])
