import json
from bs4 import BeautifulSoup

# Read the HTML file
with open("data/sites/cardht/cardht_list_page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

products = []
for product in soup.select('.wc-block-grid__product'):
    # Product name
    name_tag = product.select_one('.wc-block-grid__product-title')
    name = name_tag.get_text(strip=True) if name_tag else None
    # Product image
    img_tag = product.select_one('img')
    image = img_tag['src'] if img_tag else None
    # Product price (current price is in <ins>, fallback to <span> if not on sale)
    price_tag = product.select_one('ins .woocommerce-Price-amount')
    if not price_tag:
        price_tag = product.select_one('.woocommerce-Price-amount')
    price = price_tag.get_text(strip=True) if price_tag else None
    # Product rating
    rating_tag = product.select_one('.star-rating')
    rating = None
    rating_count = None
    if rating_tag:
        rating = rating_tag.get('aria-label')
        # Try to extract number of ratings
        count_tag = rating_tag.select_one('span.rating')
        if count_tag:
            rating_count = count_tag.get_text(strip=True)
    products.append({
        "name": name,
        "image": image,
        "price": price,
        "rating": rating,
        "rating_count": rating_count
    })

with open("data/sites/cardht/cardht_list_page.json", "w", encoding="utf-8") as f:
    json.dump(products, f, indent=2, ensure_ascii=False)

print("Analysis complete. Data saved to data/sites/cardht/cardht_list_page.json") 