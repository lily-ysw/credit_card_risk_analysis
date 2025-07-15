import json
from bs4 import BeautifulSoup
import os

base_dir = os.path.dirname(__file__)
html_path = os.path.join(base_dir, "hssza6r6_list_page.html")
json_path = os.path.join(base_dir, "hssza6r6_list_page.json")

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
products = []

for li in soup.select("ul.products li.product"):
    name_tag = li.select_one(".woocommerce-loop-product__title")
    price_tag = li.select_one(".price .woocommerce-Price-amount")
    store_tag = li.select_one(".wcfmmp_sold_by_wrapper a.wcfm_dashboard_item_title")
    url_tag = li.select_one("a.woocommerce-LoopProduct-link")
    product = {
        "name": name_tag.get_text(strip=True) if name_tag else None,
        "price": price_tag.get_text(strip=True) if price_tag else None,
        "store": store_tag.get_text(strip=True) if store_tag else None,
        "url": url_tag["href"] if url_tag else None
    }
    products.append(product)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(products, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(products)} products to {json_path}") 