# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class AudiItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # define the fields for your item here like:
    #articletext = scrapy.Field(input_processor=MapCompose(remove_tags(), remove_whitespace()),
    #                           output_processor=TakeFirst())
    article_text = scrapy.Field()
    article_date = scrapy.Field()
    article_headline = scrapy.Field()
    article_link = scrapy.Field()
    last_update = scrapy.Field(serializer=str)
