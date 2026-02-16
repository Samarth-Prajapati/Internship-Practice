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

# Beautiful Soup

soup = BeautifulSoup(content, "html.parser")
# print(soup)

prettify_soup = soup.prettify()
# print(prettify_soup)

# Tags

title = soup.title
# print(title)

tag = title.name
# print(tag)

# Navigable String

navigable_string = soup.p.string
print(navigable_string)