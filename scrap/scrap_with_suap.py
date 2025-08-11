import requests
from bs4 import BeautifulSoup
import json

# List of URLs
urls = [
    "https://www.mayoclinic.org/diseases-conditions/gastritis/symptoms-causes/syc-20355807",
    "https://www.mayoclinic.org/diseases-conditions/celiac-disease/symptoms-causes/syc-20352220",
    "https://www.mayoclinic.org/diseases-conditions/diverticulitis/symptoms-causes/syc-20371758",
    "https://www.mayoclinic.org/diseases-conditions/irritable-bowel-syndrome/symptoms-causes/syc-20360016",
    "https://www.mayoclinic.org/diseases-conditions/pancreatitis/symptoms-causes/syc-20360227",
    "https://www.mayoclinic.org/diseases-conditions/gallstones/symptoms-causes/syc-20354214",
    "https://www.mayoclinic.org/diseases-conditions/cholecystitis/symptoms-causes/syc-20364867",
    "https://www.mayoclinic.org/diseases-conditions/intestinal-obstruction/symptoms-causes/syc-20351460",
    "https://www.mayoclinic.org/diseases-conditions/lactose-intolerance/symptoms-causes/syc-20374232",
    "https://www.mayoclinic.org/diseases-conditions/functional-dyspepsia/symptoms-causes/syc-20375709",
    "https://www.mayoclinic.org/diseases-conditions/ulcerative-colitis/symptoms-causes/syc-20353326",
    "https://www.mayoclinic.org/diseases-conditions/colon-cancer/symptoms-causes/syc-20353669",
    "https://www.mayoclinic.org/diseases-conditions/constipation/symptoms-causes/syc-20354253",
    "https://www.mayoclinic.org/diseases-conditions/food-poisoning/symptoms-causes/syc-20356230",
    "https://www.mayoclinic.org/diseases-conditions/gas-and-gas-pains/symptoms-causes/syc-20372709",
    "https://www.mayoclinic.org/diseases-conditions/viral-gastroenteritis/symptoms-causes/syc-20378847",
    "https://www.mayoclinic.org/diseases-conditions/kidney-stones/symptoms-causes/syc-20355755",
    "https://www.mayoclinic.org/diseases-conditions/menstrual-cramps/symptoms-causes/syc-20374938",
    "https://www.mayoclinic.org/diseases-conditions/mittelschmerz/symptoms-causes/syc-20375122",
    "https://www.mayoclinic.org/diseases-conditions/shingles/symptoms-causes/syc-20353054",
    "https://www.mayoclinic.org/diseases-conditions/diarrhea/symptoms-causes/syc-20352241",
    "https://www.mayoclinic.org/diseases-conditions/endometriosis/symptoms-causes/syc-20354656",
    "https://www.mayoclinic.org/diseases-conditions/abdominal-aortic-aneurysm/symptoms-causes/syc-20350688",
    "https://www.mayoclinic.org/diseases-conditions/appendicitis/symptoms-causes/syc-20369543",
    "https://www.mayoclinic.org/diseases-conditions/peptic-ulcer/symptoms-causes/syc-20354223",
    "https://www.mayoclinic.org/diseases-conditions/crohns-disease/symptoms-causes/syc-20353304"
]

def scrape_article(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # Extract title and paragraphs
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "No title found"
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        article_text = " ".join(paragraphs)

        return {
            "url": url,
            "title": title,
            "content": article_text
        }
    except Exception as e:
        return {"url": url, "error": str(e)}

# Collect data
data = [scrape_article(url) for url in urls]

# Save to JSON format (single file, array of objects)
with open("medical_articles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Scraping complete. Saved to medical_articles.json")

