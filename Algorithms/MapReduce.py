"""
Lambda - anonymous function
    lambda arguments : expression

lambda value: value*2

map(func, *iterables)
Map - paradigm, which maps(apply) passed function on to iterable data structure

    pets = ['a', 'b', 'c', 'd', 'd']
    list(map(str.upper, pets)) -- without list returns map object

    circles = [1.1281, 12.1212098, -299912.8]
    print(list(map(round, circles, range(1, 3))))

filter(func, iterable)
Filter - requires the function to return boolean values (true or false).
         Then passes each element in the iterable through the function,
        "filtering" away those that are false.

    dromes = ["demigod", "rewire", "madam", "freer", "anutforajaroftuna", "kiosk"]
    list(filter(lambda word: word == word[::-1], dromes))

reduce(func, iterable[, initial])
Reduce - applies a function of two arguments cumulatively to the elements of an iterable,
         optionally starting with an initial argument.

    data = [2, 3, 4, 5, 6, 7, 8, 9]
    print(reduce(lambda x, y: x - y, data))
"""

# # Map
# pets = ['a', 'b', 'c', 'd', 'd']
# print(list(map(str.upper, pets)))
#
# # circles = [1.1281, 12.1212098, -299912.8]
# # print(list(map(round, circles, range(1, 3))))
#
# # Map and custom zip
# pets = ['dog', 'cat', 'rabbit', 'owl']
# weights = [1.1281, 12.1212098, -299912.8]
#
# print(zip(pets, weights))
# print(list(map(lambda pet, weight: (pet, weight), pets, weights)))
#
#
# # Filter
# numbers = [1, 2, -12, 0, 12, 1e10]
# print(list(filter(lambda number: number > 0, numbers)))
#
# # Find a polindrome
# dromes = ["demigod", "rewire", "madam", "freer", "anutforajaroftuna", "kiosk"]
# palindromes = list(filter(lambda word: word == word[::-1], dromes))
# print(palindromes)
#
# # Reduce
# data = [2, 3, 4, 5, 6, 7, 8, 9]
# print(reduce(lambda x, y: x - y, data))



"""
Realisation of MAP REDUCE paradigm.
"""

# Map stage

import sys
from operator import itemgetter

for line in sys.stdin:
    line = line.strip()
    words = line.split()

    for word in words:
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is 1
        print ('%s\t%s' % (word, 1))


# Reduce stage
current_word = None
current_count = 0
word = None

for line in sys.stdin:
    line.strip()
    word, count = line.split('\t', 1)

    try:
        count = int(count)
    except ValueError:
        continue
    if current_word == word:
        current_count += count
    else:
        if current_word:
            print('%s\t%s' % (current_word, current_count))
        current_count = count
        current_word = word

if current_word == word:
    print('%s\t%s' % (current_word, current_count))
