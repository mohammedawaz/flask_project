import requests
from bs4 import BeautifulSoup
import wikipedia

def scrape_google(query):
    from googlesearch import search
    urls = list(search(query, num_results=3))
    results = []
    for url in urls:
        try:
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text() for p in paragraphs)
            results.append(text[:1000])  # limit to avoid overload
        except Exception as e:
            results.append(str(e))
    return results

def scrape_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=5)
        return summary
    except Exception as e:
        return str(e)
