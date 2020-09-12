import os
import bs4
import codecs
import requests
from tqdm import tqdm
from pathlib import Path


"""
Notes:
- 200 target html files
"""

"""
Target json form:
[
    {
        'text': text,
        'labels': [
            [PUBDATE, NUMBER, NUMBER],
            [STATUS, NUMBER, NUMBER],
            [POSITION, NUMBER, NUMBER],
            [SURNAME, NUMBER, NUMBER],
            [NAME, NUMBER, NUMBER]
        ]
    }
]
"""

TARGET_PATH = '/Users/macbook/Documents/GitRep/PracticesForEngineers/Practices-for-Engineers/CompaniesPractices/Company12/SecondTask/dataset'  

files = [Path(TARGET_PATH, path) for path in os.listdir(TARGET_PATH)]

test_file_path = files[0]

TARGET_FILE = '/Users/macbook/Documents/GitRep/PracticesForEngineers/Practices-for-Engineers/CompaniesPractices/Company12/SecondTask/dataset/1.html'

# read file
html_file = codecs.open(TARGET_FILE, 'r', 'utf-8')
soup = bs4.BeautifulSoup(html_file, "html.parser")

# find RESULT part cause rest is the same for all pages 
attachement = soup.find_all('div', class_='publication_container')

C. other information 1. Organs

print(attachement)


