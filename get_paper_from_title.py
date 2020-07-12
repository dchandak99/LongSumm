# !pip install selenium

#importing libraries
import os
import time
import getpass
from tqdm import tqdm_notebook
from urllib.request import urlretrieve

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")
chrome_options.add_argument("--disable-notifications")
chrome_options.add_argument('--headless')
driver = webdriver.Chrome(executable_path='chrome/chromedriver')

base_url = 'https://dimsum.eu-gb.containers.appdomain.cloud'

paper_titles = os.listdir('LongSumm-data/extractive_summaries/talksumm_summaries')

paper_links = []
idx = 0

driver.implicitly_wait(5)

remaining_papers = []

for p_title in tqdm_notebook(paper_titles):
    driver.get(base_url) 
    search = driver.find_elements_by_xpath('//input')[0]
    search.send_keys(p_title[:-4])    
    search.send_keys(Keys.RETURN)    
    try: 
        p_link = driver.find_element_by_xpath('//*[contains(@href, ".pdf")]').get_attribute('href')
    except NoSuchElementException:
        remaining_papers.append(p_title[:-4])
    paper_links.append(p_link)
    time.sleep(1)

len(paper_links)

len(remaining_papers)

import pickle

with open('LongSumm-data/extractive_summaries/paper_links.pkl', 'wb') as file:
    pickle.dump(paper_links, file)

with open('LongSumm-data/extractive_summaries/remaining_papers.pkl', 'wb') as file:
    pickle.dump(remaining_papers, file)

if not os.path.exists('LongSumm-data/extractive_summaries/papers/'):
    os.mkdir('LongSumm-data/extractive_summaries/papers/')

paper_links

papers_not_downloaded = []

for i in tqdm_notebook(range(len(paper_links))):
    try:
        urlretrieve(paper_links[i], 'LongSumm-data/extractive_summaries/papers/'+paper_titles[i][:-4]+".pdf")
    except OSError:
        papers_not_downloaded.append(paper_links[i])


