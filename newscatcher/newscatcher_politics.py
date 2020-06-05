# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:12:10 2020

@author: victo
"""

#### necessary libraries ####
from newscatcher import Newscatcher, describe_url, urls

nc = Newscatcher(website = 'nytimes.com')
results = nc.get_news()

# results.keys()
# 'url', 'topic', 'language', 'country', 'articles'

# Get the articles
articles = results['articles']

first_article_summary = articles[0]['summary']
first_article_title = articles[0]['title']