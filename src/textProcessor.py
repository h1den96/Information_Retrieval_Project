# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# import re
# import json
# 
# # Κατεβάζουμε τα δεδομένα της NLTK αν χρειάζεται
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# 
# # Κείμενο προς προεπεξεργασία
# #text = "Αυτό είναι ένα παράδειγμα κειμένου με διάφορους ειδικούς χαρακτήρες, όπως !@#$%^&*() και κάποιες κοινές λέξεις λέξεις."
# 
# with open('./wikipedia_articles.json', 'r') as file:
#     articles_data = json.load(file)
# 
# tokens = []
# for article in articles_data:
#     article['content'] = re.sub(r'[^A-Za-zΑ-Ωα-ωΆ-Ώά-ώ\s]', '', article['content'])
#     tokens.append(word_tokenize(article['content']))
# 
#     # Αφαίρεση stop words (για ελληνικά)
#     stop_words = set(stopwords.words('greek'))
#     tokens = [word for word in tokens if word.lower() not in stop_words]
# 
#     # Stemming (προαιρετικό, περισσότερο για αγγλικά)
#     stemmer = PorterStemmer()
#     stemmed_tokens = [stemmer.stem(word) for word in tokens]
# 
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
# 
#     # Αφαίρεση ειδικών χαρακτήρων, κρατώντας ελληνικούς και λατινικούς χαρακτήρες (με τόνους και χωρίς)
#     #text = re.sub(r'[^A-Za-zΑ-Ωα-ωΆ-Ώά-ώ\s]', '', text)
# 
# # Tokenization
# #tokens = word_tokenize(text)
# 
# 
# 
# # Τελικό καθαρισμένο κείμενο
# cleaned_text = ' '.join(lemmatized_tokens)
# print("Καθαρισμένο Κείμενο:", cleaned_text)


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import json

# Downloading NLTK data if needed
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load JSON file
with open('./wikipedia_articles.json', 'r') as file:
    articles_data = json.load(file)

# Initialize preprocessing tools
stop_words = set(stopwords.words('greek'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Initialize list to hold tokens for each article
articles_tokens = []

for article in articles_data:
    # Ensure 'content' key exists
    if 'content' in article:
        # Step 1: Clean the content
        content_cleaned = re.sub(r'[^A-Za-zΑ-Ωα-ωΆ-Ώά-ώ\s]', '', article['content'])
        
        # Step 2: Tokenize the cleaned content
        article_tokens = word_tokenize(content_cleaned)
        
        # Step 3: Remove stopwords
        filtered_tokens = [word for word in article_tokens if word.lower() not in stop_words]
        
        # Step 4: Apply stemming (optional for English)
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        
        # Step 5: Apply lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
        
        # Store tokens for the current article
        articles_tokens.append(lemmatized_tokens)

# Print tokens for each article separately
article_num = 1
for article_tokens in articles_tokens:
    print(f"\nArticle {article_num} contents:\n{' '.join(article_tokens)}\n")
    article_num += 1
