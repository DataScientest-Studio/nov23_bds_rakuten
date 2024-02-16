from bs4 import BeautifulSoup as bs
import requests
import re

def get_data_from_rakuten(soup):
    designation = soup.find("span",attrs={'data-qa': "productTitle"}).string
    designation = designation.replace('\n', ' ').replace('\r', '').strip()

    description = soup.find("div",attrs={'id': "prd_information"})
    description = " ".join([info.text for info in description])
    description = description.replace('\n', ' ').replace('\r', '').strip()

    image_url = soup.find("a",attrs={'class': "prdMainPhoto"}).img.get('data-frz-src')
    return designation, description, image_url
    
def scrap(url):
    url = re.sub(r'\.html\?.*', '.html', url)
    request = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'})
    print(request.url)
    soup = bs(request.content, "html.parser")
    return get_data_from_rakuten(soup)