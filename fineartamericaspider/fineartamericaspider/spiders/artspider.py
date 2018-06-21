#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 23:35:00 2018

@author: adam
"""

from fineartamericaspider.items import FineartamericaspiderItem

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class ArtSpider(CrawlSpider):
    name = "artspider"
#    start_urls = ["https://fineartamerica.com/shop/prints?page=2"]
#    start_urls = ["https://fineartamerica.com/shop/prints/paintings?page=2"]
    
    start_urls = ["https://fineartamerica.com/shop/prints/paintings/abstract", 
                  "https://fineartamerica.com/shop/prints/paintings/contemporary", 
                  "https://fineartamerica.com/shop/prints/paintings/nature", 
                  "https://fineartamerica.com/shop/prints/paintings/drawing", 
                  "https://fineartamerica.com/shop/prints/paintings/wild",
                  "https://fineartamerica.com/shop/prints/paintings/wildlife",
                  "https://fineartamerica.com/shop/prints/paintings/impressionism",
                  "https://fineartamerica.com/shop/prints/paintings/landmark",
                  "https://fineartamerica.com/shop/prints/paintings/portrait",
                  "https://fineartamerica.com/shop/prints/paintings/still+life",
                  "https://fineartamerica.com/shop/prints/paintings/city",
                  "https://fineartamerica.com/shop/prints/paintings/surrealism",
                  "https://fineartamerica.com/shop/prints/paintings/sunset",
                  "https://fineartamerica.com/shop/prints/paintings/actor",
                  "https://fineartamerica.com/shop/prints/paintings/food+and+beverage",
                  "https://fineartamerica.com/shop/prints/paintings/politician",
                  "https://fineartamerica.com/shop/prints/paintings/transportation",
                  "https://fineartamerica.com/shop/prints/paintings/abstract+nature",
                  "https://fineartamerica.com/shop/prints/paintings/abstract+impression",
                  "https://fineartamerica.com/shop/prints/paintings/abstract+expressionism"]
    
#    start_urls = ['https://fineartamerica.com/shop/prints/paintings/celebrity',
#                  'https://fineartamerica.com/shop/prints/photographs/celebrity',
#                  'https://fineartamerica.com/shop/prints/drawings/celebrity',
#                  'https://fineartamerica.com/shop/prints/digital+art/celebrity',
#                  'https://fineartamerica.com/shop/prints/mixed+media/celebrity',
#                  'https://fineartamerica.com/shop/prints/athlete']
    allowed_domains = ['fineartamerica.com']
#    rules = (Rule(LinkExtractor(allow=('/prints',)), callback="parse_page"),)
    rules = (Rule(LinkExtractor(allow=(), restrict_xpaths=('//a[@class="buttonbottomnext"]',)), callback="parse_page", follow= True),)
        
    def parse_page(self, response):
        img = response.css(".imageprint").xpath("@src")
        imageURL = img.extract()
#        for imageURL in imageURLs:
        yield FineartamericaspiderItem(image_urls=imageURL)
#        next = response.css(".pagelistdiv").xpath("a[contains(., 'Next')]")
#        yield scrapy.Request(next.xpath("@href").extract_first(), self.parse)