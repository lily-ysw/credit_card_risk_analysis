import os
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
import time

# Path to Tor Browser's Firefox binary
# Update this path if your Tor Browser is installed elsewhere
TOR_PATH = "/Applications/Tor Browser.app/Contents/MacOS/firefox"

# Ensure the output directory exists
os.makedirs("imperialk", exist_ok=True)

# Set up Firefox options to use Tor Browser
options = Options()
options.binary_location = TOR_PATH
options.headless = False  # Set to True to run without opening a window

# Set up Tor's SOCKS proxy (Tor service default is 9050)
options.set_preference("network.proxy.type", 1)
options.set_preference("network.proxy.socks", "127.0.0.1")
options.set_preference("network.proxy.socks_port", 9050)
options.set_preference("network.proxy.socks_remote_dns", True)

# Start the browser
print("Starting Tor Browser with Selenium...")
driver = webdriver.Firefox(
    service=Service(GeckoDriverManager().install()),
    options=options
)

try:
    url = "http://imperialk4trdzxnpogppugbugvtce3yif62zsuyd2ag5y3fztlurwyd.onion/index.html#visa"
    print(f"Navigating to {url} ...")
    driver.get(url)
    time.sleep(10)  # Wait for the page to load (increase if needed)
    print("\n--- PAGE SOURCE START ---\n")
    print(driver.page_source)
    print("\n--- PAGE SOURCE END ---\n")

    # Save the page source to a file
    with open("imperialk/imperialk_list_page.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    print("Page source saved to imperialk/imperialk_list_page.html")
finally:
    driver.quit()
    print("Browser closed.") 