import requests
import bs4
import time
import re
from urllib.parse import urljoin


def scrape_wikipedia_cs_portal(max_pages=1000):
    base_url = "https://en.wikipedia.org/wiki/Portal:Computer_programming"
    visited_urls = set()
    to_visit = [base_url]
    corpus = []
    cs_pattern = re.compile(
        r'(computer science|algorithm|programming|data structure|software|database|operating system|artificial intelligence|machine learning)',
        re.IGNORECASE)

    while to_visit and len(visited_urls) < max_pages:
        url = to_visit.pop(0)
        if url in visited_urls:
            continue

        try:
            print(f"Scraping: {url}")
            response = requests.get(url)
            if response.status_code != 200:
                continue

            visited_urls.add(url)
            soup = bs4.BeautifulSoup(response.text, 'html.parser')

            # Extract main content and filter out non-relevant sections
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                continue

            # Remove unwanted elements
            for element in content_div.select('table, .navbox, .vertical-navbox, .infobox, .sidebar'):
                element.extract()

            # Get paragraphs
            paragraphs = content_div.find_all('p')
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 100 and cs_pattern.search(text):  # Only keep substantial CS-related paragraphs
                    corpus.append(text)

            # Find more CS-related links
            if len(visited_urls) < max_pages:
                links = content_div.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    if href.startswith('/wiki/') and ':' not in href and not href.startswith('/wiki/File:'):
                        if cs_pattern.search(href) or cs_pattern.search(link.get_text()):
                            full_url = urljoin('https://en.wikipedia.org', href)
                            if full_url not in visited_urls:
                                to_visit.append(full_url)

            # Be nice to Wikipedia servers
            time.sleep(1)

        except Exception as e:
            print(f"Error processing {url}: {e}")

    # Save corpus
    with open("cs_knowledge_corpus.txt", "w", encoding="utf-8") as f:
        for paragraph in corpus:
            f.write(paragraph.strip() + "\n")

    print(f"Corpus created with {len(corpus)} paragraphs from {len(visited_urls)} pages")
    return corpus


# Run the scraper
cs_corpus = scrape_wikipedia_cs_portal(max_pages=500)