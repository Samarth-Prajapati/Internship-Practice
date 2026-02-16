import requests
import pandas as pd
from bs4 import BeautifulSoup

def load_page(page_number):
    """
    Load all the pages from the provided url.
    Parameters
    ----------
    num - Number of tables to load

    Returns - Data from URL
    -------
    """

    print(f"===== Loading Page {page_number} =====\n")

    url = f"https://www.scrapethissite.com/pages/forms/?page_num={page_number}"

    try:
        response = requests.get(url)
        print(f"Page Response Status Code = {response.status_code}.\n")

        soup = BeautifulSoup(response.content, "html.parser")
        print("Beautiful Soap Loaded Successfully.\n")
        return soup

    except Exception as e:
        print("Failed Loading Page,", e)

def load_table_header(soup):
    """
    Load table header.
    Parameters
    ----------
    soup - BeautifulSoup Object

    Returns - Header from Table
    -------
    """

    print(f"===== Loading Table's Header =====\n")

    header_data = []

    try:
        table = soup.select_one("table.table")
        header = [th.get_text(strip = True) for th in table.select("tr")[0].select("th")]
        header_data.append(header)

        print("Header Fetched from Table Successfully.\n")
        return header_data

    except Exception as e:
        print("Failed Loading Table's Header,", e)

def load_table_data(soup):
    """
    Load all table data.
    Parameters
    ----------
    soup - BeautifulSoup Object

    Returns - Data from Table
    -------
    """

    print(f"===== Loading Table's Data =====\n")

    data = []

    try:
        table = soup.select_one("table.table")

        rows = table.select("tr")[1:]
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.select("td")]
            data.append(cols)

        print("Data Fetched from Table Successfully.\n")
        return data

    except Exception as e:
        print("Failed Loading Table's Data,", e)

def main():
    """
    Load url and scrap data from tables.
    Returns
    -------
    """

    print("===== Scraping Started Successfully =====\n")

    data = []
    header = []

    for page in range(1, 25):
        soup = load_page(page)

        if page == 1:
            header = load_table_header(soup)

        table_data = load_table_data(soup)

        for i in table_data:
            data.append(i)

    print("\nCreating DataFrame.\n")
    df = pd.DataFrame(data, columns = header)
    print(df)
    print("\nCreated DataFrame.\n")

    df.to_csv("practice.csv", index = False)
    print("\nCreated CSV File Successfully.\n")

if __name__ == "__main__":
    main()
