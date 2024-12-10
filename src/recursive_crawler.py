import requests
from bs4 import BeautifulSoup
import json
import time

def fetch_content_recursive(url, visited, max_depth, current_depth=1, counter=None):
    if counter is None:
        counter = [1]
    if url in visited or current_depth > max_depth:
        return []

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        visited.add(url)
        title = soup.find('h1', {"id": "firstHeading"}).get_text(strip=True) if soup.find('h1', {
            "id": "firstHeading"}) else "No title"
        content = " ".join([p.get_text(separator=" ").strip() for p in soup.find_all('p') if p.get_text(strip=True)])

        print(f"{'  ' * (current_depth - 1)}- Processing article at depth {current_depth}: '{title}' (URL: {url})")

        articles = [{"id": counter[0], "title": title, "content": content}]
        counter[0] += 1

        new_links = [
            a['href'] for a in soup.find_all('a', href=True)
            if a['href'].startswith('/wiki/') and ':' not in a['href']
        ]

        for link in new_links:
            full_url = f"https://en.wikipedia.org{link}"

            if full_url not in visited and "Main_Page" not in full_url:
                articles.extend(fetch_content_recursive(full_url, visited, max_depth, current_depth + 1, counter))

        return articles

    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []


def collect_articles(url_file, max_depth=2):
    with open(url_file, 'r') as file:
        urls = [line.strip().strip('"') for line in file if line.strip()]

    all_articles = []
    visited = set()

    for i, url in enumerate(urls, start=1):
        print(f"\nStarting to process root URL {i}: {url}")
        all_articles.extend(fetch_content_recursive(url, visited, max_depth))
        time.sleep(1)

    return all_articles


def save_to_json(data, filename="wikipedia_articles.json"):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    url_file = "urls.txt"

    max_depth = 2

    articles_data = collect_articles(url_file, max_depth=max_depth)
    save_to_json(articles_data)
