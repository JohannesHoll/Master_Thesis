# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:12:10 2020

@author: victo
"""

#### necessary libraries ####
from newscatcher import Newscatcher, describe_url, urls

nc = Newscatcher(website = 'nytimes.com', topic = 'finance')

results = nc.get_news()
articles = results['articles']
nc.print_headlines()

#headlines = results['articles'][0]['headline']
print(len(articles))
title = results['articles'][0]['title']

summary = results['articles'][0]['summary']

published = results['articles'][0]['published']

print(summary)



