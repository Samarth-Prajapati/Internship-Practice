from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd

url = "https://www.scrapethissite.com/pages/forms/?page_num=1"

def create_driver():
    """
    Create chrome driver
    Returns - chrome driver
    -------
    """

    options = Options()
    options.add_argument("--start-maximized")
    chrome_driver = webdriver.Chrome(options = options)

    return chrome_driver

def open_page(chrome_driver):
    """
    open url using chrome driver
    Parameters
    ----------
    chrome_driver - chrome driver

    Returns - None
    -------
    """

    chrome_driver.get(url)

def extract_table(chrome_driver):
    """
    Extract table data
    Parameters
    ----------
    chrome_driver - chrome driver

    Returns - extracted data
    -------
    """

    rows = chrome_driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")
    header_table = chrome_driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")
    table_data = []
    table_header = []

    for row in rows[1:]:
        cols = row.find_elements(By.CSS_SELECTOR, "td")
        cols_text = [col.text.strip() for col in cols]
        table_data.append(cols_text)

    for head in header_table:
        cols = head.find_elements(By.CSS_SELECTOR, "th")
        cols_text = [col.text.strip() for col in cols]
        table_header.append(cols_text)
        break

    return table_data, table_header

def paginate(chrome_driver, page = 24):
    """
    Paginate through all pages
    Parameters
    ----------
    chrome_driver - chrome driver
    page - number of pages

    Returns - list of all pages data
    -------
    """

    base_url = "https://www.scrapethissite.com/pages/forms/"
    all_data, header_data = [], []

    for page in range(1, page + 1):
        chrome_driver.get(f"{base_url}?page_num{page}")
        table_data, header_data = extract_table(driver)
        all_data.extend(table_data)

    return all_data, header_data

if __name__ == "__main__":
    driver = create_driver()
    data, header = paginate(driver)
    df = pd.DataFrame(data, columns = header)
    df.to_csv("practice_selenium.csv", index = False)
