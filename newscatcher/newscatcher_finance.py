# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:12:10 2020

@author: victo
"""

#### necessary libraries ####
from newscatcher import Newscatcher, describe_url, urls

nc = Newscatcher(website = 'reuters.com', topic = 'politics')

results = nc.get_news()
articles = results['articles']
print(articles)
nc.print_headlines()

#headlines = results['articles'][0]['headline']
print(len(articles))
#title = results['articles'][0]['title']

n = 0
summ = []
#for n in articles:  
summary = articles[0]['summary']
#summ.append(summary)
  #  n += 1 
    
#published = results['articles'][0]['published']

    #print(summary)

print(summary)



