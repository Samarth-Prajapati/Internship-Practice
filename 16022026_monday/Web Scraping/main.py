import requests
from bs4 import BeautifulSoup

web = requests.get("https://www.tutorialsfreak.com/")
# print(web)

status_code = web.status_code
# print(status_code)

content = web.content
# print(content)

url = web.url
# print(url)

soup = BeautifulSoup(content, "html.parser")
# print(soup)

