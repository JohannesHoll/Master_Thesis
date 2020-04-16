#!/usr/bin/env python
# coding: utf-8

# import of relevant libraries
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy import Selector
from scrapy.linkextractors import LinkExtractor
from fiancescraper.items import FiancescraperItem
import json
import numpy as np

#creating first spider

class FinanceNewsScraperSpider(scrapy.Spider):
    name = "newsarticles"
    
    def start_requests(self):
        start_urls = ['https://www.reuters.com/companies/DAIGn.DE/news',
        #              ]
        #start_urls = ['https://www.reuters.com/article/us-health-coronavirus-china-autos/chinas-car-market-seen-rebounding-from-virus-as-lockdowns-ease-idUSKCN21R13A',
                        ]
         #             'https://www.reuters.com/companies/BMWG.DE/news'        
        urls = start_urls
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse_newspage)
            #yield scrapy.Request(url=url, callback=self.parse_article)
            
    def parse_newspage(self, response):
        #links = response.xpath('//a[contains(@href,"/article/")]/text()').extract() #extract article name
        links = response.xpath('//a[contains(@href,"/article/")]/@href').extract() #extract hyperlink
        print(links)
        with open('article_links.json', 'w') as fp:
            json.dump([link for link in links], fp)
        for url in links:
            yield response.follow(url = url,callback = self.parse_article)    
        #filepath = 'DC_links1.csv'
        #with open(filepath, 'w') as f:
        #    f.writelines([link + '\n' for link in links])


    def parse_article(self, response):
        #links = response.xpath('//*[contains(@class,"ArticleHeader_headline")]/text()').extract() # extract article headline 
        #links = response.xpath('//*[contains(@class,"ArticleHeader_date")]/text()').extract() # extract article date
        #links = response.xpath('//div[@class="StandardArticleBody_body"]//p/text()').extract()
        links = response.xpath('//div[@class="StandardArticleBody_body"]//p/text()').extract()
        #joinedlinks = response.urljoin(links)     
        #for sel in response.xpath('//div[@class="StandardArticleBody_body"]'):
        #    item = FiancescraperItem()
        #    item['articel_text'] = sel.xpath('//p/text()').extract()
        #    print(item)
        l = [link for link in links]
        print(l)
        print(type(links))
        #print(len(links))
        #print(type(l))
        counter=1
        with open('article_intent%s.json' % counter, 'w') as fp:
            json.dump(l, fp, indent = 6)
            counter += 1
        #ch_titles_ext = [t.strip() for t in links]
        # Store this in our dictionary
        #dc_dict[links] = ch_titles_ext
        #filepath = 'article_intent.csv'
        #with open(filepath, 'w') as f:
        #    f.writelines([link + '\n' for link in links])

#item['name'] = sel.xpath('//a/text()').extract()
#item['link'] = sel.xpath('//a/@href').extract()            
#dont forget class ArticleHeader_date


