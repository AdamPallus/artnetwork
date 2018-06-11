#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 23:35:00 2018

@author: adam
"""

from fineartamericaspider.items import FineartamericaspiderItem
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class ArtSpider(CrawlSpider):
    name = "FineartAmerica-spider"
#    start_urls = ["https://fineartamerica.com/shop/prints?page=2"]
#    start_urls = ["https://fineartamerica.com/shop/prints/paintings?page=2"]
    start_urls = ["https://fineartamerica.com/shop/prints/paintings/abstract", 
                  "https://fineartamerica.com/shop/prints/paintings/contemporary", 
                  "https://fineartamerica.com/shop/prints/paintings/nature", 
                  "https://fineartamerica.com/shop/prints/paintings/drawing", 
                  "https://fineartamerica.com/shop/prints/paintings/wild"]
    allowed_domains = ['fineartamerica.com']
#    rules = (Rule(LinkExtractor(allow=('/prints',)), callback="parse_page"),)
    rules = (Rule(LinkExtractor(allow=(), restrict_xpaths=('//a[@class="buttonbottomnext"]',)), callback="parse_page", follow= True),)
        
    def parse_page(self, response):
        img = response.css(".imageprint").xpath("@src")
        imageURL = img.extract()
        yield FineartamericaspiderItem(image_urls=imageURL)
#        next = response.css(".pagelistdiv").xpath("a[contains(., 'Next')]")
#        yield scrapy.Request(next.xpath("@href").extract_first(), self.parse)