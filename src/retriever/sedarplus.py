import logging
import requests
import selenium.webdriver as webdriver
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


class SedarPlusLoader():
    
    def __init__(self, logger=None):
        """
        Initialize the SedarPlusLoader class.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.year = "2025"
        self.url = "https://ceo.ca/sedar"
        self.companies = ["TSK", "LUM", "CVE", "ABRA"]
        self.options = Options()
        # self.options.add_argument("--headless")
        self.options.add_argument("--start-maximized")
        self.service = Service(executable_path="/usr/local/bin/chromedriver")
        self.driver = webdriver.Chrome(service=self.service, options=self.options)

    def _scroll_up_and_reload(self):
        wait = WebDriverWait(self.driver, 20)
        try:
            # Wait for the chat container to be present
            chat_container = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".posts-container"))
            )
            
            print("Found chat container, waiting for content to stabilize...")
            time.sleep(5)  # Allow some time for initial messages to load
            
            # Move to the chat container
            actions = ActionChains(self.driver)
            actions.move_to_element(chat_container).perform()
            
            # Find the center of the chat container
            chat_location = chat_container.location
            chat_size = chat_container.size
            center_x = chat_location['x'] + chat_size['width'] // 2
            center_y = chat_location['y'] + chat_size['height'] // 2
            
            # Move cursor to the center of the chat container
            actions.move_by_offset(center_x, center_y).perform()
            
            # Number of scroll attempts
            scroll_attempts = 10
            for i in range(scroll_attempts):
                # Scroll up in the chat container
                self.driver.execute_script(
                    "arguments[0].scrollTop = 0;", chat_container
                )
                # Alternative scrolling methods if the above doesn't work:
                # actions.move_to_element(chat_container).scroll_by_amount(0, -500).perform()
                # driver.execute_script("arguments[0].scrollBy(0, -500);", chat_container)
                
                print(f"Scroll attempt {i+1}/{scroll_attempts}")
                time.sleep(2)  # Wait for content to load after each scroll

        except Exception as e:
            print(f"An error occurred: {e}")
    
    def _get_pdf_elements(self):
        """
        Get all PDF elements from the page.
        """
        # find all urls that contain 'pdf'
        pdf_elements = self.driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
        pdf_urls = [
            element.get_attribute("href") for element in pdf_elements
            if self.year in element.get_attribute("href")
            ]
        return pdf_urls

    def load_stock(self, stock):
        self.driver.get(self.url.format(stock))
        self.logger.info(f"Loading SedarPlus page for stock: {stock}")
        time.sleep(5)
        # self._scroll_up_and_reload()
        pdf_urls = self._get_pdf_elements()
        self.logger.info(f"Found {len(pdf_urls)} PDF URLs for stock ${stock}:")
        return pdf_urls

    def load(self) -> list:
        """
        Load the SedarPlus page and extract PDF URLs.

        Returns:
            list: A list of PDF URLs of sedar filings.
        """
        pdfs = []
        self.driver.get(self.url)
        time.sleep(5)
        pdf_urls = self._get_pdf_elements()
        pdfs.extend(pdf_urls)
        print(f"Found {len(pdfs)} PDF URLs in total.")
        self.driver.quit()
        return pdfs
        


if __name__ == "__main__":
    loader = SedarPlusLoader()
    pdfs = loader.load()
    print(pdfs)