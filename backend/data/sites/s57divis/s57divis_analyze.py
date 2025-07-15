import json
from bs4 import BeautifulSoup

# Read the HTML file
with open("data/sites/s57divis/s57divis_list_page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

products = []

# Find all product tables (class="table1") and their preceding <h3> as category
for h3 in soup.find_all('h3'):
    category = h3.get_text(strip=True)
    table = h3.find_next('table', class_='table1')
    if not table:
        continue
    for row in table.select('tbody tr'):
        tds = row.find_all('td')
        if len(tds) >= 2:
            name = tds[0].get_text(strip=True)
            price = tds[1].get_text(strip=True)
            products.append({
                'category': category,
                'name': name,
                'price': price
            })

# Save as JSON
with open("data/sites/s57divis/s57divis_list_page.json", "w", encoding="utf-8") as f:
    json.dump(products, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(products)} products and saved to data/sites/s57divis/s57divis_list_page.json") 