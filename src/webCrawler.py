import requests
from bs4 import BeautifulSoup
import json
import time

# Λίστα με URLs των σελίδων που θέλουμε να κατεβάσουμε
urls = [
    "https://en.wikipedia.org/wiki/Web_crawler",
    "https://en.wikipedia.org/wiki/Data_mining",
    "https://en.wikipedia.org/wiki/Information_retrieval"
    # Πρόσθεσε κι άλλες σελίδες εδώ
]

# Συνάρτηση για να συλλέξει το περιεχόμενο μιας Wikipedia σελίδας
def fetch_wikipedia_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Έλεγχος για επιτυχία στο αίτημα
        soup = BeautifulSoup(response.text, 'html.parser')

        # Εξαγωγή τίτλου
        title = soup.find('h1', {"id": "firstHeading"}).text

        # Εξαγωγή περιεχομένου
        paragraphs = soup.find_all('p')
        content = " ".join([p.text for p in paragraphs if p.text.strip()])

        # Δημιουργία ενός λεξικού με τα δεδομένα του άρθρου
        article_data = {
            "title": title,
            "url": url,
            "content": content
        }
        return article_data
    
    except requests.RequestException as e:
        print(f"Αποτυχία σύνδεσης με το URL: {url}. Σφάλμα: {e}")
        return None

# Συλλογή δεδομένων για όλα τα άρθρα
articles_data = []
for url in urls:
    article = fetch_wikipedia_article(url)
    if article:
        articles_data.append(article)
    # Αναμονή για να αποφύγουμε το μπλοκάρισμα από το Wikipedia
    time.sleep(1)

# Αποθήκευση των δεδομένων σε JSON
with open("wikipedia_articles.json", "w", encoding="utf-8") as file:
    json.dump(articles_data, file, ensure_ascii=False, indent=4)

print("Η συλλογή ολοκληρώθηκε και τα δεδομένα αποθηκεύτηκαν στο 'wikipedia_articles.json'")