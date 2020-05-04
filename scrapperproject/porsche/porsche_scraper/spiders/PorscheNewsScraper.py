#!/usr/bin/env python
# coding: utf-8

# import of relevant libraries
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy import Selector
from scrapy.linkextractors import LinkExtractor
from porsche_scraper.items import PorscheItem
import numpy as np
from scrapy.loader import ItemLoader
import csv

#creating spider for Porsche

class FinanceNewsScraperSpider(scrapy.Spider):
    name = "porschenewsarticles"
    
    def start_requests(self):
        start_urls = ['https://www.reuters.com/companies/PSHG_p.DE/news',
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
            
        item = PorscheItem()
        item['article_link'] = response.url
        item['article_headline'] = response.xpath('//*[contains(@class,"ArticleHeader_headline")]/text()').extract()
        item['article_date'] = response.xpath('//*[contains(@class,"ArticleHeader_date")]/text()').extract()
        item['article_text'] = response.xpath('//div[@class="StandardArticleBody_body"]//p/text()').extract()
        
        print(item)
        
        #saving data to file.
        file = 'article_intents.csv'
        file_name = open(file, 'a')

        fieldnames = ['article_link', 'article_headline','article_date','article_text'] #adding header to file
        #with open(file, 'w', newline='') as f:
        #    writer = csv.writer(f, delimiter=',')
        #    writer.writerow(fieldnames)
        #    writer
        #writer = csv.DictWriter(file_name, fieldnames=fieldnames)
        writer = csv.writer(file_name, lineterminator='\n')
        writer.writerow([item[key] for key in item.keys()])
        #writer.writerow(fieldnames)
        #writer.writeheader()
        #writer.writerow({'article_link': item['article_link'],
        #                 'article_headline': item['article_headline'],
        #                 'article_date': item['article_date'],
        #                 'article_text': item['article_text']}) #writing data into file.
        #with open('test.csv', 'a', newline ='') as file:
        #    writer = csv.writer(file, delimiter=',')
        #    writer.writerow(head for head in fieldnames)
        #    with open('test.csv', 'a', newline='') as f:
        #        writer = csv.writer(f, delimiter=',')
        #        writer.writerow([item[key] for key in item.keys()])
        