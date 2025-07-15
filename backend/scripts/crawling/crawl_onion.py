from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
import time

# Path to Tor Browser's Firefox binary
tor_path = "/Applications/Tor Browser.app/Contents/MacOS/firefox"

# Set up Firefox options to use Tor Browser
options = Options()
options.binary_location = tor_path
options.headless = False  # Set to True to run without opening a window

# Set up Tor's SOCKS proxy
profile = webdriver.FirefoxProfile()
profile.set_preference("network.proxy.type", 1)
profile.set_preference("network.proxy.socks", "127.0.0.1")
profile.set_preference("network.proxy.socks_port", 9150)
profile.set_preference("network.proxy.socks_remote_dns", True)
profile.update_preferences()

# Start the browser
driver = webdriver.Firefox(
    service=Service(GeckoDriverManager().install()),
    options=options,
    firefox_profile=profile
)

try:
    url = "http://bitcardshwnfg5ikvdgvwacotaxkiwbccrye6pbfjm7xckwl6mssq4ad.onion/list.php"
    driver.get(url)
    time.sleep(10)  # Wait for the page to load (increase if needed)
    print(driver.page_source)  # Print the HTML content
except Exception as e:
    print(f"Failed to fetch the page: {e}")
finally:
    driver.quit() 