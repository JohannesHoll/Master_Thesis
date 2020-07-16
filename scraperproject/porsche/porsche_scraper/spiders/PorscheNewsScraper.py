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
from datetime import datetime

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
            
        item = PorscheItem()
        item['article_link'] = response.url
        item['article_headline'] = response.xpath('//*[contains(@class,"ArticleHeader_headline")]/text()').extract()
        item['article_date'] = response.xpath('//*[contains(@class,"ArticleHeader_date")]/text()').extract()
        item['article_text'] = response.xpath('//div[@class="StandardArticleBody_body"]//p/text()').extract()
        
        print(item)
        
        #saving data to file.
        path = 'news/'
        file = 'porschenews_' + str(datetime.now().strftime("%Y%m%d-%H%M")) + '.csv'
        file_name = open(path + file, 'a')

        fieldnames = ['article_link', 'article_headline','article_date','article_text'] #adding header to file

        writer = csv.writer(file_name, lineterminator='\n')
        writer.writerow([item[key] for key in item.keys()])