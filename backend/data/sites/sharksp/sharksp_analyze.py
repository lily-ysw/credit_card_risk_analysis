import json
from bs4 import BeautifulSoup

with open("data/sites/sharksp/sharksp_list_page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

products = []

# Find the main product table (Product Listing)
table = None
for h3 in soup.find_all('h3', class_='dropdown__title'):
    if 'Product Listing' in h3.get_text():
        table = h3.find_next('table')
        break

if table:
    for row in table.select('tbody tr'):
        tds = row.find_all('td')
        if len(tds) >= 6:
            products.append({
                'product': tds[0].get_text(strip=True),
                'kind': tds[1].get_text(strip=True),
                'information': tds[2].get_text(strip=True),
                'est_value': tds[3].get_text(strip=True),
                'cost_usd': tds[4].get_text(strip=True),
                'cost_btc': tds[5].get_text(strip=True)
            })

with open("data/sites/sharksp/sharksp_list_page.json", "w", encoding="utf-8") as f:
    json.dump(products, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(products)} products and saved to data/sites/sharksp/sharksp_list_page.json") 