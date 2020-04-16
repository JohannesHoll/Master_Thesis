# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from daimler_ag.items import DaimlerAgItem
from scrapy.selector import Selector
from scrapy.http import Request


# source: https://www.data-blogger.com/2016/08/18/scraping-a-website-with-python-scrapy/
start_urls = ['https://www.reuters.com/companies/DAIGn.DE/news/']
class DaimlerAgTabletsSpider(scrapy.Spider):
    name = 'daimler_ag_tablets'
    allowed_domains = ['https://www.reuters.com/companies/DAIGn.DE/news']
    start_urls = ['https://www.reuters.com/companies/DAIGn.DE/news/']
    
    
    #rules = [Rule(LinkExtractor(allow=r'/article/', canonicalize=True, unique=True),
    #              callback='parse', follow=True)]

    # Method which starts the requests by visiting all URLs specified in start_urls
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, dont_filter=True)

   # First parse returns all the links of the website and feeds them to parse2 

    def parse(self, response):
        hxs = Selector(response)
        for url in hxs.select('//a/@href').extract():
            if not (url.startswith('http://') or url.startswith('https://')):
                url = start_urls + url 
            yield Request(url, callback=self.parse2)
  
    def parse2(self, response): 
        le = [Rule(LinkExtractor(allow=r'/article/', canonicalize=True, unique=True),
                  callback='parse', follow=True)]
        for link in le.extract_links(response):
            yield {'url': link.urk}      
            
