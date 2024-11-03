import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime

def read_urls(file_path):
    with open(file_path, 'r') as file:
        urls = [line.strip().strip('"') for line in file if line.strip()]
    return urls

def fetch_wikipedia_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.find('h1', {"id": "firstHeading"}).text if soup.find('h1', {"id": "firstHeading"}) else "No title found"

        paragraphs = soup.find_all('p')
        content = " ".join([p.text for p in paragraphs if p.text.strip()]) if paragraphs else "No content found"

        summary = "No summary found"
        for paragraph in paragraphs:
            text = paragraph.text.strip()

            if len(text) > 100 and not any(word in text.lower() for word in ["light green", "dark grey", "coordinates"]):
                summary = text
                break

        sections = []
        for header in soup.find_all(['h2', 'h3']):
            section_title = header.text.strip()
            section_content = ""
            sibling = header.find_next_sibling()
            while sibling and sibling.name not in ['h2', 'h3']:
                if sibling.name == 'p':
                    section_content += sibling.text.strip() + " "
                sibling = sibling.find_next_sibling()
            if section_content:
                sections.append({"title": section_title, "content": section_content.strip()})

        last_modified_element = soup.find("li", {"id": "footer-info-lastmod"})
        last_modified = last_modified_element.text if last_modified_element else "Unknown"
        last_modified = last_modified.replace("This page was last edited on", "").strip()

        categories = [cat.text for cat in soup.select("#mw-normal-catlinks ul li")]

        references = [ref['href'] for ref in soup.select('ol.references a[href]') if ref['href'].startswith('http')]

        infobox = {}
        infobox_table = soup.find("table", {"class": "infobox"})
        if infobox_table:
            for row in infobox_table.find_all("tr"):
                if row.th and row.td:
                    key = row.th.text.strip()
                    value = row.td.text.strip()
                    infobox[key] = value

        article_data = {
            "title": title,
            "url": url,
            "summary": summary,
            "content": content,
            "sections": sections,
            "last_modified": last_modified,
            "categories": categories,
            "references": references,
            "infobox": infobox
        }
        return article_data

    except requests.RequestException as e:
        print(f"Αποτυχία σύνδεσης με το URL: {url}. Σφάλμα: {e}")
        return None

def collect_articles(url_file):
    urls = read_urls(url_file)
    articles_data = []
    for url in urls:
        article = fetch_wikipedia_article(url)
        if article:
            articles_data.append(article)
        time.sleep(1)
    return articles_data


def save_to_json(data, filename="wikipedia_articles.json"):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Τα δεδομένα αποθηκεύτηκαν στο αρχείο '{filename}'")

if __name__ == "__main__":
    start_time = time.time()
    url_file = "urls.txt"
    articles_data = collect_articles(url_file)
    save_to_json(articles_data)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Ο συνολικός χρόνος εκτέλεσης ήταν {total_time:.2f} δευτερόλεπτα")
