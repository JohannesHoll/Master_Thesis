#!/usr/bin/env python
# coding: utf-8

# import of relevant libraries
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy import Selector
from scrapy.linkextractors import LinkExtractor
from fiancescraper.items import FiancescraperItem
import numpy as np
from scrapy.loader import ItemLoader
import csv

#creating first spider

class FinanceNewsScraperSpider(scrapy.Spider):
    name = "newsarticles"
    
    def start_requests(self):
        start_urls = ['https://www.reuters.com/companies/DAIGn.DE/news',
                      ]
        
        urls = start_urls
        
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse_newspage)
                
    def parse_newspage(self, response):
        links = response.xpath('//a[contains(@href,"/article/")]/@href').extract() #extract hyperlink
        for url in links:
            yield response.follow(url = url,callback = self.parse_article)    

    def parse_article(self, response):
        #links = response.xpath('//*[contains(@class,"ArticleHeader_headline")]/text()').extract() # extract article headline 
        #links = response.xpath('//*[contains(@class,"ArticleHeader_date")]/text()').extract() # extract article date
        #links_contents = response.xpath('//div[@class="StandardArticleBody_body"]//p/text()').extract()
        #intents = response.xpath('//div[@class="StandardArticleBody_body"]//p/text()').extract()
            
        file = 'article_intents.csv'
        item = FiancescraperItem()
        item['article_link'] = response.url
        item['article_headline'] = response.xpath('//*[contains(@class,"ArticleHeader_headline")]/text()').extract()
        item['article_date'] = response.xpath('//*[contains(@class,"ArticleHeader_date")]/text()').extract()
        item['articel_text'] = response.xpath('//div[@class="StandardArticleBody_body"]//p/text()').extract()
        
        print(item)
        

        file_name = open(file, 'a') #Output_file.csv is name of output file

        fieldnames = ['article_link', 'article_headline','article_date','articel_text'] #adding header to file
        writer = csv.DictWriter(file_name, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'article_link': item['article_link'],
                         'article_headline': item['article_headline'],
                         'article_date': item['article_date'],
                         'articel_text': item['articel_text']}) #writing data into file.
        