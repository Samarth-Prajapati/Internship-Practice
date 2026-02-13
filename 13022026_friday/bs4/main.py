from bs4 import BeautifulSoup
import requests

url = "https://beautiful-soup-4.readthedocs.io/en/latest/"

response = requests.get(url).content

soup = BeautifulSoup(response, "html.parser")
print(soup.prettify())