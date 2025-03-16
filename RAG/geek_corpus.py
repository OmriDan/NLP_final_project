import requests
import bs4
import time
import json


def scrape_geeksforgeeks(max_pages=500):
    base_url = "https://www.geeksforgeeks.org"
    categories = [
        "/data-structures/",
        "/algorithm/",
        "/web-development/",
        "/database/",
        "/computer-science-projects/",
        "/computer-network/"
    ]

    visited_urls = set()
    corpus = []

    for category in categories:
        category_url = base_url + category
        to_visit = [category_url]

        while to_visit and len(visited_urls) < max_pages:
            url = to_visit.pop(0)
            if url in visited_urls:
                continue

            try:
                print(f"Scraping: {url}")
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if response.status_code != 200:
                    continue

                visited_urls.add(url)
                soup = bs4.BeautifulSoup(response.text, 'html.parser')

                # Extract article content
                article = soup.find('article')
                if article:
                    # Remove code blocks - we want explanations, not implementations
                    for code in article.select('pre, .code-block'):
                        code.extract()

                    # Get paragraphs and headings
                    content_elements = article.select('p, h2, h3')
                    current_section = ""

                    for element in content_elements:
                        if element.name in ['h2', 'h3']:
                            if current_section:
                                if len(current_section) > 200:  # Only keep substantial content
                                    corpus.append(current_section.strip())
                            current_section = element.get_text().strip() + ": "
                        else:
                            text = element.get_text().strip()
                            if text:
                                current_section += text + " "

                    # Add the last section
                    if current_section and len(current_section) > 200:
                        corpus.append(current_section.strip())

                # Find more links within the same category
                if len(visited_urls) < max_pages:
                    content_div = soup.find('div', {'class': 'content'})
                    if content_div:
                        links = content_div.find_all('a', href=True)
                        for link in links:
                            href = link['href']
                            if href.startswith('/'):  # Internal link
                                if any(cat in href for cat in categories):
                                    full_url = urljoin(base_url, href)
                                    if full_url not in visited_urls and full_url not in to_visit:
                                        to_visit.append(full_url)

                # Be respectful to the server
                time.sleep(2)

            except Exception as e:
                print(f"Error processing {url}: {e}")

    # Save corpus
    with open("geeksforgeeks_corpus.txt", "w", encoding="utf-8") as f:
        for paragraph in corpus:
            f.write(paragraph.strip() + "\n")

    print(f"Corpus created with {len(corpus)} paragraphs from {len(visited_urls)} pages")
    return corpus


# Run the scraper
geeksforgeeks_corpus = scrape_geeksforgeeks(max_pages=300)