# Get a news data from newsapi.org
# To change topic (or select topic) of the news, change q = '' part

import requests

url = ('https://newsapi.org/v2/everything?'
       'q=Culture&'
       'from=2020-01-27&'
       'sortBy=popularity&'
       'apiKey=ddb22f3ef2b54abe8af228db83421424')

response = requests.get(url)

print (response.json())