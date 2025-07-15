import json
from bs4 import BeautifulSoup
import os

base_dir = os.path.dirname(__file__)
html_path = os.path.join(base_dir, "imperialk_list_page.html")
json_path = os.path.join(base_dir, "imperialk_list_page.json")

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
products = []

# Helper to extract product info from a section
def extract_products(section, card_type):
    for prod_col in section.find_all("div", recursive=False):
        # VISA
        img = prod_col.find("img", {"src": lambda x: x and card_type in x})
        if img:
            quantity = prod_col.find(string=lambda s: s and s.strip().startswith("x"))
            card_count = None
            for tag in prod_col.find_all("p"):
                if "Card" in tag.text:
                    card_count = tag.text.strip()
                    break
            total_balance = None
            for tag in prod_col.find_all("p"):
                if "Total Balance" in tag.text:
                    total_balance = tag.text.strip()
                    break
            price = None
            for tag in prod_col.find_all("p"):
                if tag.text.strip().startswith("$"):
                    price = tag.text.strip()
                    break
            order_btn = prod_col.find("a", string=lambda s: s and "Order Now" in s)
            order_url = order_btn["href"] if order_btn else None
            products.append({
                "type": card_type,
                "quantity": quantity.strip() if quantity else None,
                "card_count": card_count,
                "total_balance": total_balance,
                "price": price,
                "order_url": order_url
            })

# VISA
visa_anchor = soup.find("a", id="visa")
if visa_anchor:
    visa_section = visa_anchor.find_parent("div", class_="clearfix")
    if visa_section:
        extract_products(visa_section, "visaprepaid")

# MasterCard
mc_anchor = soup.find("a", id="mastercard")
if mc_anchor:
    mc_section = mc_anchor.find_parent("div", class_="clearfix")
    if mc_section:
        extract_products(mc_section, "mastercard")

# Cloned cards
cloned_anchor = soup.find("a", id="cloned")
if cloned_anchor:
    cloned_section = cloned_anchor.find_parent("div", class_="clearfix")
    if cloned_section:
        extract_products(cloned_section, "card.jpg")

# PayPal Transfers
paypal_anchor = soup.find("a", id="paypal")
if paypal_anchor:
    paypal_section = paypal_anchor.find_parent("div", class_="clearfix")
    if paypal_section:
        for prod_col in paypal_section.find_all("div", recursive=False):
            # Look for transfer amount
            transfer = None
            for tag in prod_col.find_all("p"):
                if "Transfer:" in tag.text:
                    transfer = tag.text.strip()
                    break
            price = None
            for tag in prod_col.find_all("p"):
                if tag.text.strip().startswith("$"):
                    price = tag.text.strip()
                    break
            order_btn = prod_col.find("a", string=lambda s: s and "Order Now" in s)
            order_url = order_btn["href"] if order_btn else None
            if transfer and price:
                products.append({
                    "type": "paypal_transfer",
                    "transfer": transfer,
                    "price": price,
                    "order_url": order_url
                })

# Western Union Transfers
wu_anchor = soup.find("a", id="westernunion")
if wu_anchor:
    wu_section = wu_anchor.find_parent("div", class_="clearfix")
    if wu_section:
        for prod_col in wu_section.find_all("div", recursive=False):
            transfer = None
            for tag in prod_col.find_all("p"):
                if "Transfer:" in tag.text:
                    transfer = tag.text.strip()
                    break
            price = None
            for tag in prod_col.find_all("p"):
                if tag.text.strip().startswith("$"):
                    price = tag.text.strip()
                    break
            order_btn = prod_col.find("a", string=lambda s: s and "Order Now" in s)
            order_url = order_btn["href"] if order_btn else None
            if transfer and price:
                products.append({
                    "type": "western_union_transfer",
                    "transfer": transfer,
                    "price": price,
                    "order_url": order_url
                })

# Gift Cards
gift_anchor = soup.find("a", id="gift-cards")
if gift_anchor:
    gift_section = gift_anchor.find_parent("div", class_="clearfix")
    if gift_section:
        for prod_col in gift_section.find_all("div", recursive=False):
            # VISA Gift Card
            if prod_col.find(string=lambda s: s and "VISA Gift Cards" in s):
                order_btn = prod_col.find("a", string=lambda s: s and "See more" in s)
                order_url = order_btn["href"] if order_btn else None
                products.append({
                    "type": "visa_gift_card",
                    "order_url": order_url
                })
            # Amazon Gift Card
            if prod_col.find(string=lambda s: s and "Amazon Gift Card" in s):
                order_btn = prod_col.find("a", string=lambda s: s and "See more" in s)
                order_url = order_btn["href"] if order_btn else None
                products.append({
                    "type": "amazon_gift_card",
                    "order_url": order_url
                })
            # PayPal Gift Card
            if prod_col.find(string=lambda s: s and "PayPal Gift Card" in s):
                # Check if sales stopped
                stopped = prod_col.find(string=lambda s: s and "Sales stopped" in s)
                order_btn = prod_col.find("a", string=lambda s: s and "See more" in s)
                order_url = order_btn["href"] if order_btn else None
                products.append({
                    "type": "paypal_gift_card",
                    "order_url": order_url,
                    "sales_stopped": bool(stopped)
                })

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(products, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(products)} products to {json_path}") 