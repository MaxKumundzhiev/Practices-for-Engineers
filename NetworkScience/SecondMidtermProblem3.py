# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

"""Description:
Download at least 10 000 authors from the Hungarian Repository of Academic Publications (oktatas.mtmt.hu).
a) Create a network  from the author-institute relationships, where an author is connected to an institute, if he or she works for that institute!
b) Create the lists of the most important authors! One list should be based on degree, the other one should rank authors according to betweenness.
c) Determine the component size distribution of the network!
"""


#Imports
import networkx as nx
import collections
import matplotlib.pyplot as plt
from itertools import chain
import requests as re
from bs4 import BeautifulSoup

response = re.get('http://opera.stanford.edu/composers/H.html')
soup = BeautifulSoup(response.text, "lxml")
#print(soup.prettify())

body = soup.find('body')
composer_list = body.find('ul')
list(composer_list.children)
composers = list(composer_list.findAll('li', recursive=False))
composers[0].find('b').text



