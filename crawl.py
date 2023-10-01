import mysql.connector
from datetime import datetime
import json
from peewee import *
import datetime

from regex import F

mydb = MySQLDatabase(
        "pyml",
        host="127.0.0.1", 
        port=13306,
        user="root",
        passwd="root",
        charset='utf8'  # Specify the character set as UTF-8
        )

class BaseModel(Model):
    class Meta:
        database = mydb

class CrawlData(BaseModel):
    class Meta:
        table_name = 'crawl_data'
    domain = TextField()
    title = TextField()
    url = TextField()
    content = TextField(default=None)
    date = DateTimeField(default=None)
    created_at = DateTimeField(default=datetime.datetime.now)


mydb.connect()
mydb.create_tables([CrawlData])
mydb.close()

def create_article(item, is_update=False):
    query = CrawlData.select().where(CrawlData.url == item['url'])

    if (len(query) > 0):
        itemDB = query[0]
        if is_update:
            if 'title' in item:
                itemDB.title = item['title']
            if 'content' in item:
                itemDB.content = item['content']
            if 'date' in item:
                itemDB.date = item['date']
            itemDB.save()
            print('create_article:update', item['url'])
            return itemDB
        else:
            return False

    print('create_article', item['url'], len(query))
    data = CrawlData(
        domain = item['domain'],
        title = item['title'],
        content = item['content'],
        url = item['url'],
        date = item['date'] if 'date' in item else None,
        created_at = datetime.now()
    )

    # if 'tags' in item:
    #     tagsstr = json.dumps(item['tags'])
    #     data['data'] = tagsstr

    return data.save()

import re
import requests
import json
from bs4 import BeautifulSoup
import crawl
from datetime import datetime, timedelta

class CrawlThanhNien:
    base_url = 'https://thanhnien.vn'
    page = 1
    daily = False # if exist in database will kill process.

    def __init__(self):
        pass

    def get_response(self, url, params = {}):
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Allow-Methods': 'GET, HEAD, POST, PUT, DELETE, TRACE, OPTIONS, PATCH',
            'Access-Control-Allow-Headers': 'X-Real-IP,X-AGENT,Pragma,X-REFERER,X-AUTH-TOKEN,Accept-Encoding,channel,X-XSS-Protection,X-Content-Type-Options,Strict-Transport-Security,Content-Type,Authorization,Accept,Origin,User-Agent,DNT,Cache-Control,X-Mx-ReqToken,Keep-Alive,X-Requested-With,If-Modified-Since,token-id',
            'Content-Encoding': 'gzip'
        }
        # Make an HTTP GET request to the API endpoint using the requests library
        response = requests.get(url, params=params, headers=headers)
        return response
    
    def get_page_detail(self, url, params = {}):
        url = self.base_url + url
        print('get_page_detail', url, params)
        response = self.get_response(url, params)
        if response.status_code // 100 == 2:
            content_html = response.content
            print('content_html', content_html[:100])
            # Parse the HTML using BeautifulSoup
            soup = BeautifulSoup(content_html, 'html.parser')
            # Find the title element
            publishdate_elem = soup.find('div', attrs={'data-role': 'publishdate'})
            content_detail_elem = soup.find('div', attrs={'class': 'detail__cmain-main'})

            if publishdate_elem:
                publishDate = publishdate_elem.get_text()
                publishDate = publishDate.strip()
                publishDate = dateparser.parse(publishDate)
            else:
                publishDate = None

            content_detail = content_detail_elem.get_text() if content_detail_elem else None

            return {
                'date': publishDate,
                'content': content_detail
            }
        else:
            # If unsuccessful, print the status code and reason for failure
            print(f"Request failed with status code {response.status_code}: {response.reason}")
            return {}

    def crawl_html(self, url, params = {}):
        print('crawl_html', url, params)
        response = self.get_response(url, params)
        if response.status_code // 100 == 2:
            content_html = response.content

            # Parse the HTML using BeautifulSoup
            soup = BeautifulSoup(content_html, 'html.parser')

            # Find the title element
            title_elements = soup.find_all('a', class_='box-category-link-title')
            if title_elements is None or title_elements is []:
                return False

            for title_element in title_elements:
                title = title_element.get_text()
                link = title_element['href']

                print("Title:", title)
                # print("Content:", content)
                print("Link:", link)

                data_detail = self.get_page_detail(link, {})
                print('data_detail', data_detail)
                item = {
                    'domain': 'https://thanhnien.vn/kinh-te.htm',
                    'title': title,
                    'url': link,
                    'date': data_detail['date'] if 'date' in data_detail else None,
                    'content': data_detail['content'] if 'content' in data_detail else None,
                }

                if self.daily:
                    isDup = crawl.create_article(item, False)
                    if isDup is False:
                        return False
                else:
                    crawl.create_article(item, True)
        else:
            # If unsuccessful, print the status code and reason for failure
            print(f"Request failed with status code {response.status_code}: {response.reason}")
            return False

    def run(self, page = None, daily = False):
        self.daily = daily
        self.page = 1 if page is None else page
        while (True):
            isSuccess = self.crawl_html('https://thanhnien.vn/timelinelist/18549/' + str(self.page) + '.htm', {})
            if isSuccess == False:
                break
            self.page += 1

# obj = CrawlThanhNien()
# obj.run()

import csv

class CSVWriter:
    def __init__(self, csv_file_path, fieldnames):
        self.csv_file_path = csv_file_path
        self.fieldnames = fieldnames

        # Write headers if the file doesn't exist
        file_exists = False
        try:
            with open(csv_file_path, mode='x'):
                pass
        except FileExistsError:
            file_exists = True

        if not file_exists:
            with open(csv_file_path, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

    def push_item(self, data_dict):
        with open(self.csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writerow(data_dict)

    def truncate_file(self):
        try:
            with open(self.csv_file_path, mode='w', newline='') as csv_file:
                csv_file.write('')
            print(f"CSV file '{self.csv_file_path}' has been truncated.")
        except Exception as e:
            print(f"Error truncating CSV file '{self.csv_file_path}': {e}")

from math import e
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait

from time import sleep
import dateparser
import os
import pandas as pd

DRIVER_PATH = '/Users/ngocp/.pyenv/versions/3.10.3/lib/python3.10/site-packages/selenium/webdriver'

def create_driver():
    options = Options()
    options.add_argument("--window-size=1920,1200")
    options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(options=options)
    return driver

class CrawlVietStock:
    limit_page = 20
    driver = None
    driver2 = None
    links = []
    titles = []
    retry_backup = False
    source = "https://vietstock.vn/chung-khoan.htm"
    base_url = "https://vietstock.vn"

    def __init__(self, driver = None, driver2 = None, limit_page = 20) -> None:
        print('CrawlVietStock::__init__', driver, limit_page)
        if driver is None:
            print('create_driver')
            self.driver = create_driver()
        else:
            self.driver = driver

        if driver2 is None:
            self.driver2 = create_driver()
        else:
            self.driver2 = driver2

        self.driver.implicitly_wait(3)
        self.driver2.implicitly_wait(3)
        print('create_driver::driver:', self.driver)
        self.limit_page = limit_page
        self.csv_writer = CSVWriter('backup_title_url.csv', fieldnames=["title", "url"])
        pass

    def exist_record(self, url):
        query = crawl.CrawlData.select().where(crawl.CrawlData.url == url)
        return len(query) > 0

    """
        daily: True if you want to crawl only page not exist in database
        fresh_start: True if you want to truncate backup file and start from beginning
    """
    def run(self, range_date = {'start_date': None, 'end_date': None},
            daily = False, retry_backup = False,
            fresh_start = True):
        page = 1
        self.retry_backup = retry_backup
        self.driver.get(self.source)
        if fresh_start:
            self.csv_writer.truncate_file()
        else:
            url_backup = pd.read_csv('backup_title_url.csv')
            self.links = url_backup['url'].tolist()
            self.titles = url_backup['title'].tolist()
            print('self.links ', self.links[:20], self.titles[:20])
            self.get_page_detail()
            return

        if range_date['start_date'] is not None and daily is not True:
            dateRangeElem = self.driver.find_element(By.CSS_SELECTOR, 'input[name="daterange"]')
            dateRangeElem.clear()
            dateRangeElem.send_keys(range_date['start_date'] + ' - ' + range_date['end_date'])
            dateRangeElem.click()
            self.driver.execute_script("$('.AddStockCode').remove();")
            self.driver.execute_script("$('div.trending-fixed').remove();")
            self.driver.find_element(By.CSS_SELECTOR, '.range_inputs .applyBtn').click()

        isBreakLoopList = False

        while True:
            headings_a = self.driver.find_elements(By.CSS_SELECTOR, "#channel-container .single_post_text h4 a")
            print('headings_a:len', len(headings_a))
            if len(headings_a) == 0:
                break

            for heading_elem in headings_a:
                try:
                    print('heading_elem', heading_elem)
                    title = heading_elem.text
                    link_detail = heading_elem.get_attribute("href")
                    print("Title:", title)
                    print("Link:", link_detail)
                    self.csv_writer.push_item({
                        'title': title,
                        'url': link_detail
                    })

                    if daily and self.exist_record(link_detail):
                        isBreakLoopList = True
                        break
                    else:
                        self.saveContentPage(link_detail, title, not daily)

                    self.links.append(link_detail)
                    self.titles.append(title)
                except Exception as e:
                    print('run:e:', e)
                    continue

            if isBreakLoopList:
                break

            # turn off popup trending
            btnTrending = self.driver.find_elements(By.CSS_SELECTOR, '#button-trending')
            if btnTrending:
                btnTrending[0].click()

            self.driver.execute_script("$('div.trending-fixed').remove();")

            pageCSSSelector = '#content-paging .next'
            pageNext = self.driver.find_elements(By.CSS_SELECTOR, pageCSSSelector)
            if len(pageNext) != 0:
                pageNext[0].click()
            else:
                break

            sleep(3)
            if self.limit_page != -1 and page == self.limit_page:
                break
            page += 1

    def saveContentPage(self, url, title, isUpdate = False):
        self.driver2.get(url)
        try:
            WebDriverWait(self.driver2, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'article-content')))
        except Exception as e:
            print('saveContentPage:e:', e)
            return False

        contentElem = self.driver2.find_element(By.CSS_SELECTOR, '.single_post_heading')
        content = contentElem.text
        dateElem = self.driver2.find_element(By.CSS_SELECTOR, '.blog-single-head .date')

        datePublish = None
        if dateElem:
            date = dateElem.text
            datePublish = dateparser.parse(date, date_formats=['%d/%m/%Y %H:%M'])
            print('datePublish', date, datePublish)
        
        result = crawl.create_article({
            'domain': 'https://vietstock.vn/chung-khoan.htm',
            'title': title,
            'url': url,
            'date': datePublish,
            'content': content,
        }, isUpdate)

        return result


class CrawlCafeF:
    base_url = 'https://cafef.vn'
    page = 1
    daily = False # if exist in database will kill process.

    def __init__(self):
        pass

    def get_response(self, url, params = {}):
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Allow-Methods': 'GET, HEAD, POST, PUT, DELETE, TRACE, OPTIONS, PATCH',
            'Access-Control-Allow-Headers': 'X-Real-IP,X-AGENT,Pragma,X-REFERER,X-AUTH-TOKEN,Accept-Encoding,channel,X-XSS-Protection,X-Content-Type-Options,Strict-Transport-Security,Content-Type,Authorization,Accept,Origin,User-Agent,DNT,Cache-Control,X-Mx-ReqToken,Keep-Alive,X-Requested-With,If-Modified-Since,token-id',
            'Content-Encoding': 'gzip'
        }
        # Make an HTTP GET request to the API endpoint using the requests library
        response = requests.get(url, params=params, headers=headers)
        return response
    
    def get_page_detail(self, url, params = {}):
        url = self.base_url + url
        print('get_page_detail', url, params)
        response = self.get_response(url, params)
        if response.status_code // 100 == 2:
            content_html = response.content
            soup = BeautifulSoup(content_html, 'html.parser')
            # Find the title element
            publishdate_elem = soup.find('span', attrs={'data-role': 'publishdate'})
            content_detail_elem = soup.find('div', attrs={'class': 'contentdetail'})
        
            if publishdate_elem:
                publishDate = publishdate_elem.get_text()
                publishDate = publishDate.strip()
                publishDate = dateparser.parse(publishDate)
            else:
                publishDate = None

            content_detail = content_detail_elem.get_text() if content_detail_elem else None

            return {
                'date': publishDate,
                'content': content_detail
            }
        else:
            # If unsuccessful, print the status code and reason for failure
            print(f"Request failed with status code {response.status_code}: {response.reason}")
            return {}

    def crawl_html(self, url, params = {}):
        print('crawl_html', url, params)
        response = self.get_response(url, params)
        if response.status_code // 100 == 2:
            content_html = response.content

            # Parse the HTML using BeautifulSoup
            soup = BeautifulSoup(content_html, 'html.parser')

            # Find the title element
            title_elements = soup.select('.box-category-item h3 a')
            if title_elements is None or title_elements is []:
                return False

            for title_element in title_elements:
                title = title_element.get_text()
                link = title_element['href']

                print("Title:", title)
                print("Link:", link)
                data_detail = self.get_page_detail(link, {})

                item = {
                    'domain': 'https://cafef.vn/timelinelist/18831',
                    'title': title,
                    'url': link,
                    'date': data_detail['date'] if 'date' in data_detail else None,
                    'content': data_detail['content'] if 'content' in data_detail else None,
                }
                isDup = crawl.create_article(item, not self.daily)
                print('isDup', isDup)
                return isDup 
        else:
            # If unsuccessful, print the status code and reason for failure
            print(f"Request failed with status code {response.status_code}: {response.reason}")
            return False

    def run(self, page = {"from": None, "to": None}, daily = False):
        self.daily = daily
        page_from = 1 if page['from'] is None else page['from']
        page_to = -1 if page['to'] is None else page['to']
        while (True):
            isSuccess = self.crawl_html('https://cafef.vn/timelinelist/18831/' + str(page_from) + '.chn', {})
            if isSuccess == False or page_to == page_from:
                break
            page_from += 1


class VnexpressCrawler():
    base_url = 'https://vnexpress.net'
    page = 1
    daily = False # if exist in database will kill process.

    def __init__(self):
        pass

    def get_response(self, url, params = {}):
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Allow-Methods': 'GET, HEAD, POST, PUT, DELETE, TRACE, OPTIONS, PATCH',
            'Access-Control-Allow-Headers': """X-Real-IP,X-AGENT,Pragma,X-REFERER,X-AUTH-TOKEN,Accept-Encoding,channel,X-XSS-Protection,X-Content-Type-Options,Strict-Transport-Security,Content-Type,Authorization,Accept,Origin,User-Agent,DNT,Cache-Control,X-Mx-ReqToken,Keep-Alive,X-Requested-With,If-Modified-Since,token-id""",
            'Content-Encoding': 'gzip'
        }
        # Make an HTTP GET request to the API endpoint using the requests library
        response = requests.get(url, params=params, headers=headers)
        return response
    
    def get_page_detail(self, url, params = {}):
        if url.startswith(self.base_url) == False:
            url = self.base_url + url

        print('get_page_detail', url, params)
        response = self.get_response(url, params)
        if response.status_code // 100 == 2:
            content_html = response.content
            soup = BeautifulSoup(content_html, 'html.parser')
            # Find the title element
            publishdate_elem = soup.select_one('.header-content span.date')
            if publishdate_elem:
                publishDate = publishdate_elem.get_text()
                publishDate = publishDate.strip()
                publishDate = dateparser.parse(publishDate)
            else:
                publishDate = None

            content_detail_elem = soup.select_one('article.fck_detail')

            content_detail = ''
            if content_detail_elem:
                content_detail = content_detail_elem.get_text()

            return {
                'date': publishDate,
                'content': content_detail
            }
        else:
            # If unsuccessful, print the status code and reason for failure
            print(f"Request failed with status code {response.status_code}: {response.reason}")
            return {}

    def crawl_html(self, url, params = {}):
        print('start:crawl_html:', url, params)
        response = self.get_response(url, params)
        if response.status_code // 100 == 2:
            content_html = response.content

            # Parse the HTML using BeautifulSoup
            soup = BeautifulSoup(content_html, 'html.parser')

            # Find the title element
            title_elements = soup.select('.title-news a')
            if title_elements is None or len(title_elements) == 0:
                return False

            for title_element in title_elements:
                title = title_element.get_text()
                link = title_element['href']

                print("Title:", title)
                print("Link:", link)
                data_detail = self.get_page_detail(link, {})
                item = {
                    'domain': 'https://vnexpress.net/kinh-doanh',
                    'title': title,
                    'url': link,
                    'date': data_detail['date'] if 'date' in data_detail else None,
                    'content': data_detail['content'] if 'content' in data_detail else None,
                }

                if self.daily:
                    isDup = crawl.create_article(item, False)
                    if isDup is False:
                        return False
                else:
                    crawl.create_article(item, True)

        else:
            # If unsuccessful, print the status code and reason for failure
            print(f"Request failed with status code {response.status_code}: {response.reason}")
            return False

    def run(self, page = {"from": None, "to": None}, daily = False):
        self.daily = daily
        page_from = 1 if page['from'] is None else page['from']
        page_to = -1 if page['to'] is None else page['to']
        while (True):
            isSuccess = self.crawl_html('https://vnexpress.net/kinh-doanh-p' + str(page_from), {})
            # break
            if isSuccess == False or page_to == page_from:
                break
            page_from += 1