from bs4 import BeautifulSoup
import requests

def load_bs4(url):
    """
    Parameters
    ----------
    url - Website URL for scrapping

    Returns - soup
    -------
    """

    response = requests.get(url).content
    bs4 = BeautifulSoup(response, "html.parser")
    return bs4

def load_all_data(bs4, tag):
    """

    Parameters
    ----------
    bs4 - BeautifulSoup object
    tag - Tag object to be scraped

    Returns - tag data
    -------
    """

    tags_data = bs4.find_all(tag)
    return tags_data

soup = load_bs4("https://beautiful-soup-4.readthedocs.io/en/latest/")
# print(soup.prettify())

h1_tag = load_all_data(soup, "h1")
# for i in h1_tag:
#     print(i.text)